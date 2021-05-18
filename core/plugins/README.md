# TRTorch Plugins

A library for plugins (custom layers) used in a network. This component of TRTorch library builds a separate library called `libtrtorch_plugins.so`.

On a high level, TRTorch plugin library interface does the following :

- Uses TensorRT plugin registry as the main data structure to access all plugins.

- Automatically registers TensorRT plugins with empty namepsace.  

- Automatically registers TRTorch plugins with `"trtorch"` namespace.

Here is the brief description of functionalities of each file

- `plugins.h` - Provides a macro to register any plugins with `"trtorch"`  namespace.
- `register_plugins.cpp` - Main registry class which initializes both `libnvinfer` plugins and TRTorch plugins (`Interpolate` and `Normalize`)
- `impl/interpolate_plugin.cpp` - Core implementation of interpolate plugin. Uses pytorch kernels during execution.
- `impl/normalize_plugin.cpp` - Core implementation of normalize plugin. Uses pytorch kernels during execution.

### Converter for the plugin
A converter basically converts a pytorch layer in the torchscript graph into a TensorRT layer (in this case a plugin layer).
We can access a plugin via the plugin name and namespace in which it is registered.
For example, to access the Interpolate plugin, we can use
```
auto creator = getPluginRegistry()->getPluginCreator("Interpolate", "1", "trtorch");
auto interpolate_plugin = creator->createPlugin(name, &fc); // fc is the collection of parameters passed to the plugin.
```

### If you have your own plugin

If you'd like to compile your plugin with TRTorch,

- Add your implementation to the `impl` directory
- Add a call `REGISTER_TRTORCH_PLUGINS(MyPluginCreator)`  to `register_plugins.cpp`. `MyPluginCreator` is the plugin creator class which creates your plugin. By adding this to `register_plugins.cpp`, your plugin will be initialized and accessible (added to TensorRT plugin registry) during the `libtrtorch_plugins.so` library loading.
- Update the `BUILD` file with the your plugin files and dependencies.
- Implement a converter op which makes use of your plugin.

Once you've completed the above steps, upon successful compilation of TRTorch library, your plugin should be available in  `libtrtorch_plugins.so`.

A sample runtime application on how to run a network with plugins can be found <a href="https://github.com/NVIDIA/TRTorch/tree/master/examples/sample_rt_app" >here</a>
