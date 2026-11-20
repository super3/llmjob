{
  "targets": [
    {
      "target_name": "pearl_core_fp8",
      "sources": [ "pearl_fp8_core.cc" ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "conditions": [
        ["OS=='win'", {
          "libraries": [ "<(cuda_root)/lib/x64/cudart_static.lib", "<(module_root_dir)/cuda-build/pearl_fp8.lib" ],
          "msvs_settings": { "VCCLCompilerTool": { "ExceptionHandling": 1 } }
        }],
        ["OS=='linux'", {
          "libraries": [ "-L<(module_root_dir)/cuda-build", "-lpearl_fp8", "-L<(cuda_root)/lib64", "-lcudart_static", "-lrt", "-lpthread", "-ldl" ]
        }]
      ]
    }
  ],
  "variables": {
    "cuda_root%": "<!(node -p \"process.env.CUDA_PATH || '/usr/local/cuda'\")"
  }
}
