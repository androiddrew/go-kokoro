package kokoro

import (
	"errors"
	"fmt"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

func sessionOptions(c Config) (_ *ort.SessionOptions, err error) {
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, o.Destroy())
		}
	}()
	threads := c.Threads
	if threads == 0 {
		threads = 1
	}
	for _, set := range []func() error{
		func() error { return o.SetIntraOpNumThreads(threads) },
		func() error { return o.SetInterOpNumThreads(1) },
		func() error { return o.SetExecutionMode(ort.ExecutionModeSequential) },
		func() error { return o.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll) },
	} {
		if err = set(); err != nil {
			return nil, err
		}
	}
	if c.Verbose {
		if err = o.SetLogSeverityLevel(ort.LoggingLevelVerbose); err != nil {
			return nil, err
		}
	}
	if c.Provider == CUDA {
		cuda, e := ort.NewCUDAProviderOptions()
		if e != nil {
			return nil, fmt.Errorf("CUDA options: %w", e)
		}
		err = cuda.Update(map[string]string{"device_id": strconv.Itoa(c.DeviceID), "cudnn_conv_algo_search": "DEFAULT", "use_tf32": "0"})
		if err == nil {
			err = o.AppendExecutionProviderCUDA(cuda)
		}
		err = errors.Join(err, cuda.Destroy())
		if err != nil {
			return nil, fmt.Errorf("configure CUDA (requires provider libraries compatible with the installed ONNX Runtime): %w", err)
		}
	}
	return o, nil
}

// TensorInfo describes one model input or output.
type TensorInfo struct {
	Name  string  `json:"name"`
	Shape []int64 `json:"shape"`
	Type  string  `json:"type"`
}

// ModelInfo describes the loaded model's inputs and outputs.
type ModelInfo struct {
	Inputs  []TensorInfo `json:"inputs"`
	Outputs []TensorInfo `json:"outputs"`
}

func inspectModel(path string, options *ort.SessionOptions) (ModelInfo, error) {
	// v1.22.0 exposes metadata through a temporary session, not the live session.
	// This extra model load is intentional and belongs to initialization timing.
	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(path, options)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("inspect Kokoro model: %w", err)
	}
	return validateModel(inputs, outputs)
}

func validateModel(inputs, outputs []ort.InputOutputInfo) (ModelInfo, error) {
	info := ModelInfo{}
	if len(inputs) != 3 || len(outputs) != 1 {
		return info, errors.New("expected Kokoro v1.0 model with three inputs and one output")
	}
	expected := map[string]struct {
		shape []int64
		dtype ort.TensorElementDataType
	}{
		"tokens": {[]int64{1, -1}, ort.TensorElementDataTypeInt64},
		"style":  {[]int64{1, 256}, ort.TensorElementDataTypeFloat},
		"speed":  {[]int64{1}, ort.TensorElementDataTypeFloat},
		"audio":  {[]int64{-1}, ort.TensorElementDataTypeFloat},
	}
	seen := make(map[string]bool)
	for i, group := range [][]ort.InputOutputInfo{inputs, outputs} {
		for _, item := range group {
			e, ok := expected[item.Name]
			if !ok || seen[item.Name] || (item.Name == "audio") != (i == 1) || item.OrtValueType != ort.ONNXTypeTensor || item.DataType != e.dtype || len(item.Dimensions) != len(e.shape) {
				return ModelInfo{}, fmt.Errorf("incompatible Kokoro tensor: %s", &item)
			}
			for j, dim := range item.Dimensions {
				if dim != e.shape[j] {
					return ModelInfo{}, fmt.Errorf("incompatible Kokoro tensor shape: %s", &item)
				}
			}
			seen[item.Name] = true
			t := TensorInfo{item.Name, append([]int64(nil), item.Dimensions...), item.DataType.String()}
			if i == 0 {
				info.Inputs = append(info.Inputs, t)
			} else {
				info.Outputs = append(info.Outputs, t)
			}
		}
	}
	return info, nil
}
