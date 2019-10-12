ATen "native" functions are the modern mechanism for adding operators and
functions to ATen (they are "native" in contrast to legacy functions, which are bound
via TH/THC cwrap metadata).  Native functions
are declared in `native_functions.yaml` and have implementations defined
in one of the `cpp` files in this directory.

Like all ATen methods/functions, native functions are made available
from both ATen's C++ and Python APIs.  In C++, they are made available
either as methods on `Tensor` (`t.mymeth()`) and functions in the ATen
namespace (`at::myfunc()`).  In PyTorch, they are made available as
methods on `Variable` or as functions on `torch._C._FunctionBase`
(it is the user's responsibility to re-exporting these functions in
a more user-facing module.)  At the moment, only
functions which ingest `Variable` are made available; to use a function
with non-differentiable tensors, wrap your tensors with `Variable` before
passing them in.

The rest of this document describes how to implement an ATen function.

## Registering a function in `native_functions.yaml`

Every native function must have an entry in
`native_functions.yaml`.  The format can be summarized as:

```
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  variants: function, method
  dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

Each component is described in more detail below:

### `func`

```
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
```

The `func` entry is a string describing the name of the function and its type
signature.

**Argument types.** These types are permissible as ArgType:

- `Tensor`.  A `Tensor` argument translates into a C++ argument of type `const Tensor&`
  (except when the argument is "inplace"; in this case, it is simply `Tensor&`).
  A trailing `?`, as in `Tensor?`, indicates that the tensor argument is optional
  and may be omitted by passing an undefined tensor.  When a function takes multiple
  `Tensor` arguments, these tensors are assumed to be the same type (e.g.,
  if one argument is a `FloatTensor`, all other arguments are checked
  to be `FloatTensor`s.)
- Tensors of specific types.  At the moment, valid type names are:
    - `IntegerTensor` (a.k.a. `LongTensor`)
    - `BoolTensor` (a.k.a. `ByteTensor`)
    - `IndexTensor` (a.k.a. `IntTensor`)
  These type names were inherited from TH, and may be renamed soon, so
  don't commit them to memory.
- `TensorList`.  A `TensorList` argument translates into a C++ argument of type `ArrayRef<Tensor>`
  (a.k.a. `TensorList`)
- `IntList`.  `IntList` accepts an optional length specifier, e.g., `IntList[2]`, which
  has no effect in C++ but extends our Python bindings to accept a bare number, which will be
  expanded into an appropriately sized list by repeating the number.
- `int64_t`. There is no `int`; ATen policy is to use `int64_t` in the API anywhere you would
  have ordinarily passed an `int` or `size_t`.
- `double`. There is no `float`; ATen policy is to use `double` anywhere you would have used `float`.
- `bool`
- `Scalar`. `Scalar` supports binding to any numerical types from Python, including integral types,
  floating point types, and zero dimensional tensors. `int64_t` and `double` can only bind to the
  corresponding Python numerical types. However, you probably don't want to use `Scalar`. It's
  really used for binding to TH/THC code "real" types where the Python APIs you are binding to are
  actually different types. `double` and `int64_t` argument types should suffice for most algorithms.
- `Generator*`, the state for a random number generator,
- `std::array<bool,N>` (where N is `1-4`).  NB: you MUST NOT put a space after the comma, otherwise
  this argument will not parse correctly.  (If you decide to fix this, make sure you fix the
  argument parser both in ATen and in PyTorch.)
- `TensorOptions`.  Tensor options provide information about how a
  tensor should be constructed; it is most useful when you are writing a
  factory function, where you have no `Tensor` inputs and thus
  cannot otherwise determine how to construct a `Tensor`.
- `*` is a special sentinel argument, which doesn't translate into an actual
  argument, but indicates that in the Python bindings, any subsequent arguments
  must be specified as keyword arguments (and cannot be provided positionally).
- `?` is trailing question mark that annotate an argument to be an optional type, grep for
  `optional` to find some example usages. In general, most functions will not need to use
  this, but there are some cases that we want to use optional for the different types:
    - You want to pass in a `None` to a ATen function/method from Python, and handles the
      None type in the C++ side. For example, `clamp(Tensor self, Scalar? min=None, Scalar? max=None)`
      can take `None` for its `min` and `max` parameter, and do dispatch to different
      backend if one of the parameters is `None`. Optional type can accept a `None` type
      (`nullopt` in C++) from Python and use the [C++ Optional class](https://en.cppreference.com/w/cpp/utility/optional) to interact with the parameters.
    - You want a default value which is fine in Python but would cause ambiguity in C++.
      For example, `norm(Tensor self, Scalar p=2, int64_t dim, bool keepdim=false)` would
      cause ambiguity in C++ since it default args must be adjacent and `p` could not
      have a default value when `dim` does not. Therefore, we need to make `p` as a
      optional Scalar, and make `p=2` when `p` is not passed in (nullopt).
    - You want a value to default to the same value as another argument (this cannot be
      expressed in C++ default arguments).

Functions with no tensor inputs are called *factory functions*, and
are handled specially by code generation.  If your function is behaving
differently than another example, check first and see if one is a
factory while another is not.

**Argument names.** Argument names are meaningful; downstream binding code may make use of the specific
argument name you provide, and a rename of an argument name is considered a BC-breaking
change (e.g., you will probably need to update `tools/autograd/derivatives.yaml` at
least). In `native_functions.yaml`, if your function (usually functions named with 'out' affix) args
include the result Tensor, you need to call the argument `Tensor result`. And if there are more
than one result Tensors, you need to name the args `Tensor result0, Tensor result1, ...`.

TODO: Do argument names affect Python keyword arguments?

**Defaults.** Any suffix of arguments can have a default value defined;
these default values translate into C++/Python default values which
are applied when those positional arguments are not specified.

Here are the supported default values:

* Numbers (e.g., `0` or `5.0` for `int64_t`, `double` and `IntList`
  with an explicit length (e.g., `IntList[2]`)--in the case of IntList,
  a number is replicated to fill the length (e.g., `IntList[2] x=2`
  is equivalent to `IntList[2] x={2,2}`.
* Lists of numbers (e.g., `{0, 0}`) for `IntList`.
* Booleans (e.g., `true`) for `bool`.
* Empty initializer lists (e.g., `{}`) for `Tensor` (this implicitly changes
  a `Tensor` argument to accept undefined tensors).
* `nullptr` for pointer types (e.g., `Generator*`)

**Returns.** The following are permissible on Return:

Non-tuple return:
```
ReturnType [retarg0]
```

Tuple return:
```
(ReturnType [retarg0], ReturnType [retarg1], ...)
```

The following are permissible on ReturnType:
- `Tensor` and `TensorList`, which translate into the C++ types `Tensor` and `std::vector<Tensor>`,
  respectively (unless the operation is in-place, in which case the return type
  is `Tensor&`.
- A tuple of any number of `Tensor`, e.g., `(Tensor, Tensor)`, translating into
  the C++ `std::tuple<Tensor, Tensor>`.

If you need a type that is not listed in this list, it may be possible to extend ATen's
code generation to support it.  ATen's philosophy on types to support is that it supports
only simple, universal types, as well as a handful of fundamental Tensor structures
(e.g., `Tensor` and `Generator*`), because these types can be easily ported to any language
bound to ATen (in practice, C++ and Python.)

Return also supports specifying (optional) return argument names; these are useful for writing
derivatives in terms of return arguments in `tools/autograd/derivatives.yaml`.

Note that argument type modifiers such as defaults and optional are not currently supported on Return.


The declarations also support the following attributes:

### `variants`

```
variants: function, method
```

Controls whether Tensor method (`t.foo()`) or namespace Function (`at::foo()`) is
generated as a result of this declaration.  If the declaration is a method,
you must have an argument `Tensor self` at some position in the method;
in the method variant this argument will be elided from the argument
list.  For example, given the declaration `where(BoolTensor cond, Tensor self, Tensor other)`,
this generates the function `at::where(cond, self, other)` and the method
`self.where(cond, other)`.

By default, ATen generates only the function variant for a native function.
When should you also generate a method variant?  Tensor operations as methods
are appropriate for "core" Tensor operations (e.g., add, sub, etc.), but not for
more complicated neural network layers (e.g., `conv2d`) and internal functions
designed specifically for binding (e.g., `cudnn_convolution`).

### `dispatch`

```
dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

This specifies the actual name of the function you want to dispatch to, so you
can dispatch to different functions depending on whether or not you have CPU or
CUDA tensors.  Technically, it is also possible to write `dispatch: func_name`
to unconditionally dispatch to a native function whose name is different than
the name in the public ATen API, but this is generally frowned upon (just name
them the same thing!)

### `device_guard`

```
device_guard: false
```

By default, ATen code generation will generate a DeviceGuard invocation,
which will ensure that kernel code will run with the current device set
to match the device of the first Tensor argument (or first tensor of
the first TensorList argument, if the function takes a list of tensors).
For the most part, this means kernel authors do not have to worry about
setting devices.

However, in some cases, setting the device is unnecessary, because,
e.g., you call a function already manages device guard setting, or
you're a function that simply does not interact with any devices.  In
that case, code generation of the device guard can be disabled by adding
`device_guard: false` to your function definition.

**Note.** We are considering eliminating automatic generation of DeviceGuard,
in which case this field would go away.  If you have an opinion on the
matter, please write in at https://github.com/pytorch/pytorch/issues/14234

## Writing an implementation in C++

Implementations of native functions go in an appropriate C++ file in the
`native/` directory (they are organized roughly by topic, but there is no
semantic meaning to their organization aside for the `cuda` directory,
which is the only place the build system knows how to build `cu` files.)
To write a native function, you only need to write a C++
implementation (no header necessary) with a matching signature to
the generated header from the ATen metadata.  There are many
simple native functions; take a look at some of them to see what to do.

Although, for the most part, writing an ATen function is mostly writing
the algorithm you want to implement, there are some less obvious details
you should also consider.

### Will your function be automatically differentiable?

If you are writing a pair of functions `foo` and `foo_backward`, with
the intent that `foo_backward` implements the derivative of `foo`, then
your implementation of `foo` is probably not automatically differentiable:
it might make use of functions like `data_ptr()` or it dispatches differently
depending on if it's operating on CPU or CUDA tensors.  Once you write these two functions,
you will have to write an entry correlating them together in
`tools/autograd/derivatives.yaml`.

However, in some situations, you can write a function in ATen and it
will be automatically differentiated!  This can be the case if the function implementation
only calls other operations which are themselves differentiable.  In this
case, you don't have to write an entry in `tools/autograd/derivatives.yaml`.

### Can it handle being passed Variables?

The biggest subtlety of writing an ATen implementation is the fact that
`Tensor` is not a "final" class: your implementation may be passed objects
which inherit from `Tensor` (in particular, the `Variable` subclass
implements automatic differentiation in PyTorch.)  This has some
direct consequences on valid implementations:

* Never create a `Tensor` directly (e.g., `at::CPU` or `at::CUDA`), as a
  caller will be expecting to get `Variable`s out if it passes `Variable`.
  Instead, create tensors using the `options()` of one of the input
  tensors.  E.g., `at::empty(sizes, input.options())` or
  `at::ones(input.options().dtype(kByte))`, if you need
  a different scalar type.

* If you need to call other ATen functions, be sure to qualify the call
  with `at::`; don't call them unqualified (in the `at::native` namespace).
  Using the qualified name ensures that your invocation gets dispatched to
  the `Variable` (which may be overridden to behave differently than
  simply dispatch to `at::native`).

These are not hard and fast rules: in particular, if you explicitly define
a derivative for a function, it will only ever be called with `Tensor`
arguments.  However, it is considered good style to abide by these rules,
since code written in this style is more robust.

NB: There is one downside to following the `at::` qualification rule, which
is that if you know that you will only ever be called with `Tensor`, a
direct `at::native` call will be more efficient (as it avoids a dynamic
dispatch).

### How to handle broadcasting?

Unlike our legacy TH bindings, ATen native functions do not automatically
handle broadcasting; you will have to insert the necessary broadcasting
calls yourself.

When writing broadcasting code, we obey the convention that `op` is
broadcasting, while `s_op` (with the `s_` prefix) is not broadcasting.  The
relationship is best seen by an example of how you would implement broadcasting
addition out of non-broadcasting addition:

```
#include <ATen/ExpandUtils.h>

Tensor add(const Tensor& self, const Tensor& other) {
  Tensor b_self, b_other;
  std::tie(b_self, b_other) = expand_outplace(self, other, "add");
  return s_add(b_self, b_other);
}

Tensor s_add(const Tensor& self, const Tensor& other) {
  // non-broadcasting implementation of addition
}
```

For inplace operations, the convention looks like this:

```
Tensor& add_(Tensor& self, const Tensor& other) {
  Tensor b_other = expand_inplace(self, other, "add_");
  return s_add_(self, b_other);
}

Tensor& s_add_(Tensor& self, const Tensor& other) {
  // non-broadcasting implementation of inplace addition
}
```

### Undefined tensor conventions

By default, `Tensor` arguments to ATen functions are always defined, unless
you explicitly specified that an undefined tensor was permissible by writing
`Tensor?` or `Tensor? x={}`, the latter one is needed when you have to assign a default value in C++ (e.g. in the middle of other parameters with default values).

The rules for returning undefined Tensors are a bit more subtle, but there
is only one case you have to remember:

* If the function in question is a backward function which accepts a
  `std::array<bool,N> output_mask` argument, you MUST return an undefined
  `Tensor` at every tuple position `i` for which `output_mask[i]` is false, otherwise

* You MUST NOT return an undefined tensor.

The most common situations where you might be tempted to return undefined tensors
are when:

- You have a forward function that may return a buffer if training is enabled, but does not
  return the buffer in inference mode.  In this case, just return an appropriately
  typed zero-size tensor.

- You have a backward function where the gradient for an input is zero.  In this case, you
  are expected to create a zero-filled tensor of appropriate size to return for this input.
  To get the shape, it may be helpful to take a `TensorGeometry` of the input to use.

### Debugging tips

If you build ATen and get a linker error, that probably means you copy-pasted
the C++ definition of your function incorrectly.  Double check your `Tensor`
arguments, and make sure you wrote `const Tensor&` in your signature.
