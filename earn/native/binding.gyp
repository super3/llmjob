{
  "targets": [
    {
      "target_name": "pearl_core",
      "sources": [ "src/pearl_core.cc" ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "conditions": [
        ["OS=='win'", {
          "libraries": [ "<(cuda_root)/lib/x64/cudart.lib", "<(module_root_dir)/cuda-build/pearl_cuda.lib" ],
          "include_dirs": [ "<(cuda_root)/include" ],
          "msvs_settings": { "VCCLCompilerTool": { "ExceptionHandling": 1 } }
        }],
        ["OS=='linux'", {
          "libraries": [ "-L<(cuda_root)/lib64", "-lcudart", "-L<(module_root_dir)/cuda-build", "-lpearl_cuda" ],
          "include_dirs": [ "<(cuda_root)/include" ]
        }]
      ]
    }
  ],
  "variables": {
    "cuda_root%": "<!(node -p \"process.env.CUDA_PATH || '/usr/local/cuda'\")"
  }
}
