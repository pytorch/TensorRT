# buildifier: disable=module-docstring
DependencyCollectionInfo = provider(doc = "", fields = ["type"])

collection_types = ["default", "jetpack"]

def _impl(ctx):
    _type = ctx.build_setting_value
    if _type not in collection_types:
        fail(str(ctx.label) + " build setting allowed to take values {" +
             ", ".join(collection_types) + "} but was set to unallowed value " +
             _type)

    return DependencyCollectionInfo(type = _type)

dep_collection = rule(
    implementation = _impl,
    build_setting = config.string(flag = True),
)
