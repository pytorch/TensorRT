# Torch-TensorRT Plugins

A library for plugins (custom layers) used in a network. This component of Torch-TensorRT library builds a separate library called `libtorchtrt_plugins.so`.

On a high level, Torch-TensorRT plugin library interface does the following :

- Uses TensorRT plugin registry as the main data structure to access all plugins.

- Automatically registers TensorRT plugins with empty namepsace.

- Automatically registers Torch-TensorRT plugins with `"torch_tensorrt"` namespace.

Here is the brief description of functionalities of each file

- `plugins.h` - Provides a macro to register any plugins with `"torch_tensorrt"`  namespace.
- `register_plugins.cpp` - Main registry class which initializes both `libnvinfer` plugins and Torch-TensorRT plugins (`Interpolate` and `Normalize`)
- `impl/interpolate_plugin.cpp` - Core implementation of interpolate plugin. Uses pytorch kernels during execution.
- `impl/normalize_plugin.cpp` - Core implementation of normalize plugin. Uses pytorch kernels during execution.

### Converter for the plugin
A converter basically converts a pytorch layer in the torchscript graph into a TensorRT layer (in this case a plugin layer).
We can access a plugin via the plugin name and namespace in which it is registered.
For example, to access the Interpolate plugin, we can use
```
auto creator = getPluginRegistry()->getPluginCreator("Interpolate", "1", "torch_tensorrt");
auto interpolate_plugin = creator->createPlugin(name, &fc); // fc is the collection of parameters passed to the plugin.
```

### If you have your own plugin

If you'd like to compile your plugin with Torch-TensorRT,

- Add your implementation to the `impl` directory
- Add a call `REGISTER_TORCHTRT_PLUGIN(MyPluginCreator)`  to `register_plugins.cpp`. `MyPluginCreator` is the plugin creator class which creates your plugin. By adding this to `register_plugins.cpp`, your plugin will be initialized and accessible (added to TensorRT plugin registry) during the `libtorchtrt_plugins.so` library loading.
- Update the `BUILD` file with the your plugin files and dependencies.
- Implement a converter op which makes use of your plugin.

Once you've completed the above steps, upon successful compilation of Torch-TensorRT library, your plugin should be available in  `libtorchtrt_plugins.so`.

A sample runtime application on how to run a network with plugins can be found <a href="https://github.com/pytorch/TensorRT/tree/master/examples/torchtrt_runtime_example" >here</a>
