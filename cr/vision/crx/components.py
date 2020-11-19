


def step(unit, inputs, outputs=None):
    if isinstance(inputs, str):
        inputs = [inputs]
    is_multi_output = isinstance(outputs, list) or isinstance(outputs, tuple)
    is_multi_output = is_multi_output and len(outputs) > 0
    is_single_output = isinstance(outputs, str)
    def f(context):
        in_data = []
        for name in inputs:
            value = getattr(context, name)
            in_data.append(value)
        result = unit(*in_data)
        if outputs is None:
            # Nothing to do.
            return context
        if is_single_output:
            setattr(context, outputs, result)
        elif is_multi_output:
            for i, name in enumerate(outputs):
                setattr(context, name, result[i])
        return context
    return f
