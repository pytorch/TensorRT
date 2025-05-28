# buildifier: disable=module-docstring
DepSrcInfo = provider(doc = "", fields = ["type"])

src_types = ["archive", "whl", "local"]

def _impl(ctx):
    src = ctx.build_setting_value
    if src not in src_types:
        fail(str(ctx.label) + " build setting allowed to take values {" +
             ", ".join(src_types) + "} but was set to unallowed value " +
             src)

    return DepSrcInfo(type = src)

dep_src = rule(
    implementation = _impl,
    build_setting = config.string(flag = True),
)
