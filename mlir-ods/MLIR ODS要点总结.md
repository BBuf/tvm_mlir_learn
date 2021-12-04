# 前言
在[【从零开始学深度学习编译器】十二，MLIR Toy Tutorials学习笔记一](https://mp.weixin.qq.com/s/jMHesvKmAUU5dYH0WznulA) 中提到MLIR是通过Dialect来统一各种不同级别的IR，即负责定义各种Operation（算子）。然后对Dialect和Operation的定义又是通过TabelGen规范构造的，通过TableGen驱动MLIR的Operation定义也被称作ODS（ Operation Definition Specification) 。我们目前只是简单认识了Toy Tutorials的Dialect和Operation是如何通过ODS定义的，但对ODS本身的语法以及一些限制都没有太多了解，这就导致在看一些相关工程的Operation定义时时常陷入迷惑，不知道某个字段是什么含义，或者说自定义Op的时候的应当如何声明操作数和Attr（举个例子，要将卷积的groups参数设置为可选的属性，应该怎么做）。

因此这篇文章将基于MLIR的ODS文档来讲解ODS中的一些要点，帮助我们更好的了解和上手MLIR。我会把官方文档中需要注意的点拆成一些小的要点。下面文章中提到的TableGen和ODS不做特别区分，ODS中的语法也就是TableGen语法。这里介绍的要点在OneFlow对接MLIR时都或多或少用到了，感兴趣的可以对照着看看OneFlow的这部分源码。`https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/include/OneFlow/OneFlowOps.td `。


# 1. 为什么要使用ODS来定义Operation
在MLIR中要定义Operation支持用C++直接定义以及基于ODS框架定义两种方法。使用C++直接定义要求我们继承基类Op的一些构造方法并重写，对于每一个Op都要写一段C++代码。可以想到这样做整个系统的Op定义部分会非常冗余，产生大量可重复代码并且可读性也会比较差。如果基于ODS来定义Operation，我们只需要将Op定义按照ODS的规范统一写到一个`td`文件中，然后使用MLIR提供的代码生成工具自动生成Operation的C++定义，这种完全auto codegen的方式非常优雅的实现了Operation定义并且需要用户操心的东西（也就是ODS的语法规范）更加直观。

ODS是MLIR定义Operation的不二选择，因此我们有必要学习ODS的语法规范。

# 2. TableGen语法
一个TableGen文件（以`.td`结尾）包含以下一些语法：
- TableGen `class` 类似于C++的class，可以作为模板或者基类去派生子类。
- TableGen `def` 类似于C++的对象。以用一个TableGen `class`的特化来声明，例如，`def MyDef: MyClass<...>;`，也可以单独使用`def MyDef;`。它不能用作模板，也不能作为基类去派生子类。
- TableGen `dag` 是一种专门用于有向无环图元素的类型。一个`dag`类型带有一个操作符和零个或者多个参数。语法形如(`operator arg0, arg1, argN`.)，其中`operator`可以是任意的TableGen `def`。参数可以是任何东西，包括`dag`本身。我们可以将名称附加到操作符和参数上，如(`MyOp:$op_name MyArg:$arg_name`)。

想了解更多TableGen支持的类型和表达式可以点这个链接：https://llvm.org/docs/TableGen/ProgRef.html。

# 3. Operation定义
MLIR定义了几个公共的结构用于帮助定义Operation，并通过`TableGen backend : OpDefinitionsGen`提供它们的语义。这些公共结构在文件`OpBase.td`中定义。主要包括：

- `Op`类：这是定义Operation时使用的主要结构。在特化该类时，通过下述结构的帮助，指定与Operation有关的所有事实。
- `Dialect`类：归属于同一个逻辑组的Operation会被放置在同一个Dialect下。Dialect包含了方言等级信息。
- `OpTrait`类及其子类：它们用于指定Operation的特殊属性和约束，包括Operation是否具有副作用、Op的输出是否与输入具有相同的形状等。
- `ins/outs`标记：这是`OpDefinitionsGen`后端内置的两个特殊标记，分别引导操作数(operands)/属性(attributes)、结果(results)的定义。
- `TypeConstraint`类及其子类：它们用于指定对操作数(operands)或结果(results)的约束。一个值得注意的子类是`Type`，它代表通用C++类型的约束。
- `AttrConstraint`类及其子类：它们用于指定对属性(attributes)的约束。一个值得注意的子类是`Attr`，它代表值为通用类型的属性的约束。

一个Operation是通过特化`Op`类定义的，特化后的`Op`类包含它需要的所有字段的具体内容。举个例子，`tf.AvgPool`定义如下：

```cpp
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoSideEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvertDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
```

下面描述一下定义一个Operation所需的所有字段。有关支持的字段的完整列表，请参阅`Op`类的定义(就是`OpBase.td`)。

- **Operation name** : 就是Operation的名字，比如TensorFlow Dialect中的`tf.Add`。
- **Operation documentation** : Operation的文档描述，包含`summary`和`description`两种，大家看下就懂，不多说。
- **Operation arguments** ： Operation的参数，一个Operation有两种参数一种是`operands`即操作数，一种是`attributes`属性参数。其中属性参数又分为`Natural attributes`和`Derived attributes`两种，前者为自然属性必须指定比如卷积的输出通道数，后者为派生属性比如输出Tensor的形状。

操作数和属性都在`dag`类型的`arguments`中被指定，以`ins`引导：

```cpp
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>,
  ...
);
```

这里`<type-constraint>`是一个来自`TypeConstraint`类层次的TableGen `def`。与此类似的，`<attr-constraint>`是一个来自`AttrConstraint`类层次的TableGen `def`。在Constraints章节有更多详细内容。


- **可变操作数**。定义一个可变操作数，需要用`Variadic<...>`把`TypeConstraint`包起来。通常，Operation是没有可变操作数或者只有一个可变操作数。对于后一种情况，可以通过静态可变操作数的定义很容易的推导出动态可变操作数。但是，如果一个Operation有多个可变长度操作数(可选的或可变长度的)，在没有来自该操作的进一步信息的情况下，就不可能将动态操作数归因于相应的静态可变长度操作数定义。因此，需要用`SameVariadicOperandSize`或`AttrSizedOperandSegments`特征来表明所有的可变长度操作数都有与之对应的动态值。
- **可选操作数**。定义一个可选操作数，需要用`Optional<...>`把`TypeConstraint`包起来。解释和可变操作数一样。
- **可选属性**。定义一个可选属性，需要使用`OptionalAttr<...>` 把`AttrConstraint `包起来。
- **带默认值的可选属性**。使用`DefaultValuedAttr<..., "...">`把`AttrConstraint`包起来。`DefaultValuedAttr`的第二个参数应该是包含C++默认值的字符串。举个例子，一个单精度浮点默认值需要被指定为`“0.5f”`，一个整型数组的默认值需要被指定为`"{1， 2， 3}"`。
- **限制属性(Confining attributes)**。`Confined`作为一种通用机制被提供，以帮助对值类型带来的属性约束进行进一步建模。可以通过`Confined`将较为原始的约束组合成为复杂约束。举个例子，一个`32bit`的整型最小值为10，可以被表示为`Confined<I32Attr, [IntMinValue<10>]>`。还有一些其它例子，比如`IntMinValue<N>`：指定一个大于等于`N`的整型属性等等。
- **Operation results**：类似操作数，结果使用`tag`类型的`results`声明，使用`outs`引导。

```cpp
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

- 还有**Operation regions**和**Operation successors**目前我还没用过，暂时不清楚应用场景。

- **Op的特征和约束(Operation traits and constraints)**：特征是影响语法或语义的Operation属性。MLIR C++的各种特征在`mlir::OpTrait`命名空间中。Operation的特征、接口或者约束涉及多个操作数/属性/结果时，要作为`Op`类的第二个模板参数传入。它们都需要继承于`OpTrait`类。详见Constraints章节。

# 4. Operation自动生成的默认构建方法
定义了Operation之后，我们怎么构建呢？ 每一个Operation，都会基于Operation的参数和Operation的返回值自动生成一些`builers`。举个例子，给出如下的Operation定义：

```cpp
def MyOp : ... {
  let arguments = (ins
    I32:$i32_operand,
    F32:$f32_operand,
    ...,

    I32Attr:$i32_attr,
    F32Attr:$f32_attr,
    ...
  );

  let results = (outs
    I32:$i32_result,
    F32:$f32_result,
    ...
  );
}
```

下面的`builders`被产生：

```cpp
// All result-types/operands/attributes have one aggregate parameter.
// 所有 结果类型/操作数/属性都集合为一个聚合参数。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ArrayRef<Type> resultTypes,
                  ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are of mlir::Attribute types.
// 每一个 结果类型/操作数/属性 都是一个独立的参数。属性参数为 mlir::Attribute 类型
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are raw values unwrapped with mlir::Attribute instances.
// (Note that this builder will not always be generated. See the following
// explanation for more details.)
// 每一个 结果类型/操作数/属性 都是一个独立的参数。
// 属性参数是未经 mlir::Attribute 实例包装的原始值。
// (注意，该构建器并不总是生成。详见下列解释获得更多细节。)
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  APInt i32_attr, StringRef f32_attr, ...);

// Each operand/attribute has a separate parameter but result type is aggregate.
// 每一个 操作数/属性 都是一个独立的参数。但是结果全部集合为了一个聚合类型。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ArrayRef<Type> resultTypes,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...);

// All operands/attributes have aggregate parameters.
// Generated if return type can be inferred.
// 这个构建器只有在返回值类型能够被推断出的情况下，才会生成。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands, ArrayRef<NamedAttribute> attributes);

// (And manually specified builders depending on the specific op.)
```

上面的代码注释翻译已经解释了这些builder的不同之处。并且可能还存在一些其它的builder，请参考https://mlir.llvm.org/docs/OpDefinitions/#run-mlir-tblgen-to-see-the-generated-content 这里的文档进行查看。



# 5. 自定义builder方法
假设上面生成的C++代码中构造方法没有我们所期待的，这个时候我们就需要自定义builder方法。比如：

```cpp
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val)>
  ];
}
```

`builders`字段是添加到Op类的自定义构建器列表。在这个例子中，我们提供了一个方便的builer，它接受浮点值而不是属性。在使用TableGen `dag`的ODS中，许多函数声明都使用`ins`前缀。紧随其后的是用逗号分隔的列表，列表的每一项都是类型与带`$`前缀的名字的组合。上述定义将会转换成如下格式的builder：


```cpp
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val);
};
```

注意，这个builder有两个额外的前置参数。这些参数对于构建Operation很有用。特别地，为了能够通过该方法构建Operation，必须向`state`填充该Operation的属性，操作数，域和返回值类型。`builder`可以用于构建属于Op的任意IR对象，例如类型或嵌套操作。当类型与名字转换为C++代码时，它们应该是有效的C++结构，一个类型(在Op的命名空间中)与一个标识符(例如，`class`不是一个有效标识符)。可以在ODS中直接提供builder的实现，使用如下TableGen的代码块：

```cpp
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

`$_builder`和`$_state`这两个特殊参数等效于`builder`和`state`。`ins`部分中的参数可以被直接使用，比如`val`。builer的c++代码实现会通过替换ODS中的特殊变量来完成，要保证builder ODS实现的其他部分是有效的C++结构。虽然对代码大小没有限制，但我们鼓励只在ODS中内联较短定义的builder，而将定义较长的builder的定义放在C++文件中。最后，如果某些参数需要默认值，可以使用 `CArg` 定义它们以包装类型和此值，如下所示：

```cpp
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins CArg<"float", "0.5f">:$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

转换后的C++代码中，默认参数只在声明中出现，而不会在定义中出现，这符合C++要求。

```cpp
/// Header file.
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val = 0.5f);
};

/// Source file.
MyOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
            float val) {
  state.addAttribute("attr", builder.getF32FloatAttr(val));
}
```

# 6. 声明指令格式（Declarative Assembly Format）
Operation的声明指令格式可以在与Operation操作数、属性等匹配的声明性字符串中指定。具有表达需要解析以构建Operation的附加信息能力：


```cpp
def CallOp : Std_Op<"call", ...> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}
```

主要由三部分组成：

- **Directives：指令**。指令是一种带有可选参数的内置函数。可用的指令有`attr-dict`，`attr-dict-with-keyword`，`operands`，`ref`等等。
- **字面值(Literals)** 。字面值是用``包裹起来的键值或者标点符号。下列是有效的标点符号集合：`:, ,, =, <, >, (, ), {, }, [, ], ->, ?, +, *` 。`\n`标点符号有另起一行的效果。如下：

```cpp
let assemblyFormat = [{
  `{` `\n` ` ` ` ` `this_is_on_a_newline` `\n` `}` attr-dict
}];
```

```cpp
%results = my.operation {
  this_is_on_a_newline
}
```
内容为空的字面量可用于删除隐式插入某些字面量元素后的空格。例如`)`或者`]`等等。举个例子，`]`可能出现在输出output的末尾，但它并不是格式中的最后一个元素，在这个例子里可以使用 `"]``"`删除掉后续的空格。

- **Variables(变量)**。变量是注册在Operation上的实体，例如Operation的参数(属性或操作数)，域，结果，后继者，等等。在`CallOp`中，变量代表`$callee`和`$args`。属性变量将显示其各自的值类型。除非其值的类型可以构造，在这种情况下，属性变量的值类型可以省略。


# 7. 自定义指令(Custom Directives) & 可选组(Optional Groups)
声明指令格式规范在格式化一个Operation的时候能够处理大部分的普通场景。对于那些想要在格式中指定Operations的某一部分的Op，声明式语法是不支持的，这个时候可以尝试使用自定义指令。

在某些情况下，Operations可能具有“可选”信息，例如 属性或一组空的可变参数操作数。 在这些情况下，可以根据此信息的存在将汇编格式的一部分标记为可选。 

这两部分比较复杂，我还没用到，所以这里不展开，感兴趣请看官方文档。

# 8. 类型推断
格式的一项要求是操作数和结果的类型必须始终存在。在某些情况下，可以通过类型约束或其他可用信息来推断变量的类型。 在这些情况下，可以从格式中省略该变量的类型。 
- **Buildable Types。可构建类型** 。一些类型约束可能只有一种表示，允许它们直接构建； 例如 `I32` 或`Index`类型。 ODS 中的类型可以通过设置 `builderCall` 字段或从 `BuildableType` 类继承来将自己标记为可构建。 
- **Trait Equality Constraints。特征等价约束**。有许多Operations具有在Operations上注册为已知类型相等特征的约束； 例如，`select` Operation的真、假和结果值通常具有相同的类型。 汇编格式可以检查这些等价的约束以辨别缺失变量的类型。 当前支持的特征有：`AllTypesMatch`、`TypesMatchWith`、`SameTypeOperands` 和 `SameOperandsAndResultType`。 
- **InferTypeOpInterface** 。实现 `InferTypeOpInterface` 的Operations可以在其汇编格式中省略其结果类型，因为可以从操作数中推断出结果类型。 
- **hasCanonicalizer**。此布尔字段指示是否已为此Operation定义规范化模式。 如果它是 `1`，那么 `::getCanonicalizationPatterns()` 应该被定义。 
- **hasCanonicalizeMethod**。当此布尔字段设置为 `true` 时，表示操作为简单的“matchAndRewrite”样式规范化模式实现了`canonicalize`方法。 如果 `hasCanonicalizer` 为 0，则实现 `::getCanonicalizationPatterns()` 的实现来调用此函数。
-  **hasFolder**。此布尔字段指示是否已为此操作定义了通用折叠规则。 如果它是 `1`，那么 `::fold()` 应该被定义。 

# 9. 额外声明
表驱动操作定义的目标之一是为每个操作自动生成尽可能多的逻辑和方法。 话虽如此，总会有无法涵盖的长尾案例。 对于这种情况，您可以使用 `extraClassDeclaration`。 `extraClassDeclaration` 中的代码将逐字复制到生成的 C++ op 类。 

请注意，`extraClassDeclaration` 是一种针对高级用户的长尾案例的机制； 对于尚未实施的广泛适用的情况，改善基础设施是可取的。 

# 10. 生成C++代码
`OpDefinitionsGen` (https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp)处理Operation定义规范文件(`.td`文件)并生成两个包含相应 C++ 代码的文件：一个用于声明，另一个用于定义。 前者通过 `-gen-op-decls` 命令行选项生成，而后者通过 `-gen-op-defs` 选项生成。

定义文件包含所有的op方法定义，可以通过定义`GET_OP_CLASSES`来包含和启用。 对于每个操作，OpDefinitionsGen 生成一个操作类和一个操作数适配器类(`operand adaptor class`)。 此外，它还包含一个以逗号分隔的所有已定义Operations的列表，可以通过定义 `GET_OP_LIST` 来包含和启用这些操作。 

- **Class name and namespaces** 。

对于每个Operation，其生成的C++类名是使用TableGen `def`为前缀的名字，并删除了Dialect前缀。第一个`_`用作分隔符。例如，对于`def TF_AddOp`，C++类名会是`AddOp`。我们移除了`TF`前缀，因为它是多个Operation作用域。其它Dialect也可以定义自己的AddOps。

生成的C++类的namespace将来自Dialect的`cppNamespace`字段。举个例子，如果一个Dialect的`Namespace`是`A::B`，那么该Dialect的Op将被放置在`namespace A { namespace B { ... } }`。 如果Dialect没有指定`cppNamespace`，我们就使用方言的名称作为命名空间。 

这意味着生成的 C++ 类的名称不一定与Operation 名称中的操作名称完全匹配。 这是为了允许灵活命名以满足编码风格要求。


- **Operand adaptors**

对于每个Operation，MLIR会自动生成一个操作数适配器。这个类解决了访问作为列表值提供的操作数而不使用“魔术“”常量的问题。 操作数适配器引用一个 `Value` 数组，并提供与Operation类中名称相同的方法来访问它们。例如，对于二元算术运算，它可以提供 `.lhs()` 来访问第一个操作数和 `.rhs()` 来访问第二个操作数。 操作数适配器类与Operation类位于同一命名空间中，类的名称由Operation类的名称后面接一个`Adaptor`组成。

操作数适配器也可以用于处理Operation的函数模板： 

```cpp
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());;
}

void process(AddOp op, ArrayRef<Value> newOperands) {
  zip(op);
  zip(Adaptor<AddOp>(newOperands));
  /*...*/
}

```

在OneFlow中，我们可以看到生成的`UserOpAdaptor`代码。里面提供了一系列接口可以访问Operation的操作数以及相关属性。

```cpp
//===----------------------------------------------------------------------===//
// ::mlir::oneflow::UserOp declarations
//===----------------------------------------------------------------------===//

class UserOpAdaptor {
public:
  UserOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  UserOpAdaptor(UserOp &op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange data_input();
  ::mlir::ValueRange ctrl_inputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr op_name();
  ::mlir::BoolAttr trainable();
  ::mlir::StringAttr device_tag();
  ::mlir::ArrayAttr device_name();
  ::mlir::IntegerAttr scope_symbol_id();
  ::mlir::ArrayAttr hierarchy();
  ::mlir::DenseIntElementsAttr operand_segment_sizes();
  ::mlir::DenseIntElementsAttr result_segment_sizes();
  ::mlir::StringAttr op_type_name();
  ::mlir::ArrayAttr input_lbn_segment_keys();
  ::mlir::ArrayAttr input_lbn_segment_sizes();
  ::mlir::ArrayAttr output_lbn_segment_keys();
  ::mlir::ArrayAttr output_lbn_segment_sizes();
  ::mlir::ArrayAttr output_lbns();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
```

# 11. 约束
约束(Constraint)是表驱动Operation定义中的一个核心概念：Operation验证和图Operation匹配都是基于满足约束。因此，Operation定义和重写规则都直接涉及写入约束。MLIR在`OpBase.td`(`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`)中定义了`Constraint`基类。一个Operation的约束可以覆盖不同的范围，可能是：

- 仅关注单个属性（例如大于 5 的 32 位整数）
- 多个操作数和结果（例如，第一个结果的形状必须与第一个操作数（可理解为Tensor）相同）
- 操作本身固有的。（例如没有副作用，参考Transpose Op消除那个案例）

我们将它们分别称为单实体约束、多实体约束和特征。 



# 前言
这一节在[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 的基础上补充完整了ODS的要点。约束和属性的定义都是MLIR中相当重要的元素，至于类型的定义个人认为了解即可，等到我们需要自定义类型的时候再仔细研究。最后MLIR的语法比较晦涩，初学者可以借助`mlir-tblgen`来辅助debug。

在这两篇文章里，我跟着MLIR的ODS规范完整走了一遍并总结了14个要点，对于每一个要点我都在OneFlow MLIR的Op定义中进行了对照，并给出了一些示例代码和位置。希望对读者入门MLIR有帮助。

# 11. 约束（这个很重要）
约束(Constraint)是表驱动Operation定义中的一个核心概念：Operation验证和图Operation匹配都是基于约束来做的。因此，Operation定义和重写规则都直接涉及写入约束。MLIR在`OpBase.td`(`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`)中定义了`Constraint`基类。一个Operation的约束可以覆盖不同的范围，可能是：

- 仅关注单个属性（例如大于 5 的 32 位整数）
- 多个操作数和结果（例如，第一个结果的形状必须与第一个操作数（可理解为Tensor）相同）
- 操作本身固有的。（例如没有副作用，参考Transpose Op消除那个案例）

我们将它们分别称为单实体约束、多实体约束和特征。这里的概念了解下即可，我觉得写新的约束是最重要的。

- **单体约束**。单体约束作用域为单个操作数，属性或结果的约束在实体的声明位置进行指定，如**Operation arguments** 和 **Operation results** 中（在[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 中总结了Operation arguments和Operation results需要注意的知识）。
- **多实体约束**。多实体约束在`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`中被建模为`PredOpTrait`类（是`OpTrait`的一个子类）。查看`OpBase.td`获取完整列表。
- **特征**。特征是Operation的内在属性，例如是否具有副作用、可交换与否、是否是终止符等。这些约束应指定为 Op 类模板参数，如[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 中第三节的Op的特征和约束(Operation traits and constraints) 所示。特征在`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`中被建模成一个`NativeOpTrait`类（`OpTrait`的一个子类）。 它们得到支持并将被翻译成相应的 C++ `mlir::OpTrait` 类。 

- **如何指定新的约束**？要写一个新的约束，我们必须为它提供一个谓词并指定一个描述名。使用`Pred`类建模的谓词是构成约束的核心。约束的谓词通常以嵌套的方式构建，有两种类型的谓词：1.`CPred`：原始的叶子节点谓词。2.复合谓词：由使用谓词组合器的子谓词组成的谓词（conjunction: `And`, disjunction: `Or`, negation: `Neg`, substitution: `SubstLeaves`, concatenation: `Concat`）。`CPred` 是构成更复杂谓词的基础。 它是TableGen 视角下的“原子”谓词，是TableGen 与C++ 之间的“接口”。 里面已经是 C++ 代码了，它会被当作不透明的字符串来处理，并带有特殊的占位符来替换。 我们可以将任何返回布尔值的 C++ 代码放在 `CPred` 中，包括计算表达式、调用函数、调用类方法等。 

为了帮助与 C++ 环境交互，提供了一些特殊的占位符来引用使用该谓词的上下文中的实体。 它们充当封闭环境的“钩子”。 这包括 `$_builder`、`$_op` 和 `$_self`：

- `$_builder`会被替换成一个`mlir::Builder`实例，以便我们可以访问常见的构建方法。
- `$_op` 会被当前的Operation替换，以便我们可以访问当前Operation的信息。
- `$_self` 会被替换为该谓词所附加的实体。 例如，`BoolAttr` 是一个包含 `CPred<"$_self.isa<BoolAttr>()">` 的属性约束。 那么对于 `BoolAttr:$attr`，`$_self` 将被 `$attr` 替换。 对于类型约束，它有点特殊，因为我们希望每个类型定义的约束自然读取，并且我们希望将类型约束直接附加到操作数/结果，`$_self` 将被操作数/结果的类型替换。 例如，对于 `F32:$operand` 中的 `F32`，它的 `$_self` 将被扩展为`operand(...).getType()`。

例如，要写一个属性 `attr` 是一个 `IntegerAttr`，在 C++ 中我们可以调用 `attr.isa<IntegerAttr>()`来实现。 这行代码也可以作为 `$_self.isa<IntegerAttr>()` 包装在 `CPred` 中，其中 `$_self` 作为特殊占位符，在扩展时由当前属性 `attr` 替换来实现相同的功能（指在Tablegen中）。

对于更复杂的谓词，我们可以将其包装在单个 `CPred` 中，也可以使用谓词组合器将它们组合起来。 例如，要写出属性 `attr` 是 32 位或 64 位整数的约束，可以将其写为：

```cpp
And<[
  CPred<"$_self.isa<IntegerAttr>()">,
  Or<[
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(32)">,
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(64)">
  ]>
]>
```
（注意，上面只是用一个熟悉的例子来展示如何使用`CPred`和谓词组合器来编写复杂的谓词。具体来说，对于整数属性，`OpBase.td`已经定义了`I32Attr`和`I64Attr`。所以我们实际上可以重用它们来编写它 `Or<[I32Attr.predicate, I64Attr.predicate]>`.)

这里再以OneFlow的一个例子来讲解一下，我们定义了一个IsGPU的约束：

```cpp
def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;
```

然后OneFlow在Transformer部分做了一个定制优化，就是将Scale和Tril这两个连续的Kernel融合成一个大的Kernel，这样可以省掉一部分内存读写的时间。但这个融合的kernel只在GPU的情况下生效，所以这个时候就需要判断当前计算图检测到的Scale和Tril这两个Operation的device是否是GPU的，就需要这个约束。FusedScaleTrilPattern这个Pass的实现如下，可以看到在最后使用了IsGPU这个约束。

```cpp
def FusedScaleTrilPattern : Pat<
  (
    OneFlow_TrilOp
    (
      OneFlow_ScalarMulOp
        $x,
        $scale_op_name,
        $scale_trainable,
        $scale_device_tag,
        $scale_device_name,
        $scale_scope_symbol_id,
        $scale_hierarchy,
        $has_int_operand,
        $has_float_operand,
        $int_operand,
        $float_operand
    ),
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;
```

这个Pass的功能就是检测到连续的Scale+Tril Operation就将这两个Operation融合成一个FusedScaleTril Operation。

如果谓词用 `CPred` 和谓词组合器一起编写非常复杂，我们也可以将其编写为普通的 C++ 函数，并使用 `CPred` 作为“调用”函数的一种方式。 例如，要验证属性 `attr` 是否具有某些属性，我们可以编写一个 C++ 函数，如：

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

然后定义Op如下：

```cpp
def HasSomeProperty : AttrConstraint<CPred<"HasSomeProperty($_self)">,
                                     "has some property">;

def MyOp : Op<...> {
  let arguments = (ins
    ...
    HasSomeProperty:$attr
  );
}
```

至于我们是否应该使用单个 `CPred` 包装整个表达式、多个带有谓词组合器的 `CPreds` 或单个 `CPred` “调用”一个函数来定义谓词，没有明确的标准。 使用 `CPred` 和谓词组合器进行定义是可取的，因为它将更多信息（而不是隐藏 C++ 函数背后的所有逻辑）公开到操作定义规范中，以便它可以潜在地驱动更多的自动生成案例。 但它需要一个很好的通用谓词库作为构建块，以避免重复，目前正在研究中。 

# 12. 属性定义（很重要+1）
属性是编译期就知道的Operation的常量。ODS 在 C++ 属性类上提供属性包装器。 MLIR 的核心 IR 库中定义了一些常见的 C++ 属性类（`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h`）。ODS 允许在 TableGen 中使用这些属性来定义Operation，可能具有更细粒度的约束。 比如`StrAttr`直接映射到`StringAttr`； `F32Attr/F64Attr` 要求 `FloatAttr` 额外具有一定的位宽。 ODS属性被定义为具有存储类型（对应于存储属性的`mlir::Attribute`类），返回类型（对应于生成的`getters`帮助函数的C++返回类型）以及在内部存储类型和帮助函数进行互转的方法。

**属性装饰器**。 有一些重要的属性适配器/装饰器/修饰符可以应用于 ODS 属性以指定常见的附加属性，如可选性、默认值等。
- `DefaultValuedAttr`：为一个属性指定默认值。
- `OptionalAttr`：将一个属性指定为可选的。
- `Confined`：`Confined`作为一种通用机制被提供，以帮助对值类型带来的属性约束进行进一步建模。可以通过`Confined`将较为原始的约束组合成为复杂约束。举个例子，一个`32bit`的整型最小值为10，可以被表示为`Confined<I32Attr, [IntMinValue<10>]>`。还有一些其它例子，比如`IntMinValue<N>`：指定一个大于等于N的整型属性等等。

**枚举属性** 。某些属性只能从预定义的enum获取值，例如，比较op的比较类型。 为了定义这些属性，ODS 提供了几种机制：`StrEnumAttr`、`IntEnumAttr` 和 `BitEnumAttr`。
- `StrEnumAttr`：每个enum case 都是一个字符串，属性在op中存储为 `StringAttr`。 
- `IntEnumAttr`：每个enum case 都是一个整数，属性在op中存储为 `IntegerType`。 
- `BitEnumAttr`：每个 enum case 都是一个位，属性在 op 中存储为 `IntegerAttr`。

所有这些 `*EnumAttr` 属性都需要通过其对应的 `*EnumAttrCase` 完全指定所有允许的情况。 有了这个，ODS 能够生成额外的验证以只接受允许的案例。 为了促进 `*EnumAttrs` 和它们的 C++ 使用者之间的交互，EnumsGen(`https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/EnumsGen.cpp`) TableGen 后端可以生成一些常见的实用程序：C++ 枚举类、用于枚举类的 `llvm::DenseMapInfo`、从/到字符串的转换函数。 这是通过 `mlir-tblgen` 的 `-gen-enum-decls` 和 `-gen-enum-defs` 命令行选项控制的。 

例如，给定下面的`EnumAttr`：

 

```cpp
def Case15: I32EnumAttrCase<"Case15", 15>;
def Case20: I32EnumAttrCase<"Case20", 20>;

def MyIntEnum: I32EnumAttr<"MyIntEnum", "An example int enum",
                           [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```
以下代码将通过 `mlir-tblgen -gen-enum-decls` 生成： 

```cpp
namespace Outer {
namespace Inner {
// An example int enum
enum class MyIntEnum : uint32_t {
  Case15 = 15,
  Case20 = 20,
};

llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t);
llvm::StringRef ConvertToString(MyIntEnum);
llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMyIntEnum() {
  return 20;
}

} // namespace Inner
} // namespace Outer

namespace llvm {
template<> struct DenseMapInfo<Outer::Inner::MyIntEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline Outer::Inner::MyIntEnum getEmptyKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getEmptyKey());
  }

  static inline Outer::Inner::MyIntEnum getTombstoneKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Outer::Inner::MyIntEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const Outer::Inner::MyIntEnum &lhs, const Outer::Inner::MyIntEnum &rhs) {
    return lhs == rhs;
  }
};
}
```

以下代码将通过 `mlir-tblgen -gen-enum-defs` 生成： 


```cpp
namespace Outer {
namespace Inner {
llvm::StringRef ConvertToString(MyIntEnum val) {
  switch (val) {
    case MyIntEnum::Case15: return "Case15";
    case MyIntEnum::Case20: return "Case20";
  }
  return "";
}

llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef str) {
  return llvm::StringSwitch<llvm::Optional<MyIntEnum>>(str)
      .Case("Case15", MyIntEnum::Case15)
      .Case("Case20", MyIntEnum::Case20)
      .Default(llvm::None);
}
llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t value) {
  switch (value) {
  case 15: return MyIntEnum::Case15;
  case 20: return MyIntEnum::Case20;
  default: return llvm::None;
  }
}

} // namespace Inner
} // namespace Outer
```

对于以下 `BitEnumAttr` 定义类似： 

```cpp
def None: BitEnumAttrCase<"None", 0x0000>;
def Bit1: BitEnumAttrCase<"Bit1", 0x0001>;
def Bit2: BitEnumAttrCase<"Bit2", 0x0002>;
def Bit3: BitEnumAttrCase<"Bit3", 0x0004>;

def MyBitEnum: BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit1, Bit2, Bit3]>;
```

我们得到：

```cpp
// An example bit enum
enum class MyBitEnum : uint32_t {
  None = 0,
  Bit1 = 1,
  Bit2 = 2,
  Bit3 = 4,
};

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
std::string stringifyMyBitEnum(MyBitEnum);
llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef);
inline MyBitEnum operator|(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline MyBitEnum operator&(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(MyBitEnum bits, MyBitEnum bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

namespace llvm {
template<> struct DenseMapInfo<::MyBitEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline ::MyBitEnum getEmptyKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getEmptyKey());
  }

  static inline ::MyBitEnum getTombstoneKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::MyBitEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::MyBitEnum &lhs, const ::MyBitEnum &rhs) {
    return lhs == rhs;
  }
};
```

```cpp
std::string stringifyMyBitEnum(MyBitEnum symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  llvm::SmallVector<llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Bit1"); val &= ~1u; }
  if (2u & val) { strs.push_back("Bit2"); val &= ~2u; }
  if (4u & val) { strs.push_back("Bit3"); val &= ~4u; }

  if (val) return "";
  return llvm::join(strs, "|");
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return MyBitEnum::None;

  llvm::SmallVector<llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<llvm::Optional<uint32_t>>(symbol)
      .Case("Bit1", 1)
      .Case("Bit2", 2)
      .Case("Bit3", 4)
      .Default(llvm::None);
    if (bit) { val |= *bit; } else { return llvm::None; }
  }
  return static_cast<MyBitEnum>(val);
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return MyBitEnum::None;

  if (value & ~(1u | 2u | 4u)) return llvm::None;
  return static_cast<MyBitEnum>(value);
}
```

在OneFlow-MLIR中同样也有枚举属性的定义用来处理OneFlow的各种数据类型，代码如下：

```cpp
#ifndef ONEFLOW_ENUMS
#define ONEFLOW_ENUMS

def OneFlow_InvalidDataType : I32EnumAttrCase<"DT_InvalidDataType", 0>;
def OneFlow_Char : I32EnumAttrCase<"DT_Char", 1>;
def OneFlow_Float : I32EnumAttrCase<"DT_Float", 2>;
def OneFlow_Double : I32EnumAttrCase<"DT_Double", 3>;
def OneFlow_Int8 : I32EnumAttrCase<"DT_Int8", 4>;
def OneFlow_Int32 : I32EnumAttrCase<"DT_Int32", 5>;
def OneFlow_Int64 : I32EnumAttrCase<"DT_Int64", 6>;
def OneFlow_UInt8 : I32EnumAttrCase<"DT_UInt8", 7>;
def OneFlow_OFRecord : I32EnumAttrCase<"DT_OFRecord", 8>;
def OneFlow_Float16 : I32EnumAttrCase<"DT_Float16", 9>;
def OneFlow_TensorBuffer: I32EnumAttrCase<"DT_TensorBuffer", 10>;

def OneFlow_DataType: I32EnumAttr<"DataType", "OneFlow Data Type enum",
  [
    OneFlow_InvalidDataType,
    OneFlow_Char,
    OneFlow_Float,
    OneFlow_Double,
    OneFlow_Int8,
    OneFlow_Int32,
    OneFlow_Int64,
    OneFlow_UInt8,
    OneFlow_OFRecord,
    OneFlow_Float16,
    OneFlow_TensorBuffer,
  ]
> {
  let cppNamespace = "::mlir::oneflow";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

#endif // ONEFLOW_ENUMS
```

我们可以观察一下它生成的enum属性声明：


```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace oneflow {
// OneFlow Data Type enum
enum class DataType : uint32_t {
  DT_InvalidDataType = 0,
  DT_Char = 1,
  DT_Float = 2,
  DT_Double = 3,
  DT_Int8 = 4,
  DT_Int32 = 5,
  DT_Int64 = 6,
  DT_UInt8 = 7,
  DT_OFRecord = 8,
  DT_Float16 = 9,
  DT_TensorBuffer = 10,
};

::llvm::Optional<DataType> symbolizeDataType(uint32_t);
::llvm::StringRef ConvertToString(DataType);
::llvm::Optional<DataType> ConvertToEnum(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDataType() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(DataType enumValue) {
  return ConvertToString(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DataType> symbolizeEnum<DataType>(::llvm::StringRef str) {
  return ConvertToEnum(str);
}

class DataTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DataType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DataTypeAttr get(::mlir::MLIRContext *context, DataType val);
  DataType getValue() const;
};
} // namespace oneflow
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::oneflow::DataType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::oneflow::DataType getEmptyKey() {
    return static_cast<::mlir::oneflow::DataType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::oneflow::DataType getTombstoneKey() {
    return static_cast<::mlir::oneflow::DataType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::oneflow::DataType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::oneflow::DataType &lhs, const ::mlir::oneflow::DataType &rhs) {
    return lhs == rhs;
  }
};
}
```

实现部分就不贴了，这里贴了过长的代码了。

# 13. 类型定义（我只是简单了解了一下）
MLIR 定义了 `TypeDef` 类层次结构，以支持根据其规范生成数据类型。 类型是通过特化 `TypeDef` 类来定义的，该类具有它所需的所有字段的具体内容。 例如，整数类型可以定义为： 


```cpp
// All of the types will extend this class.
class Test_Type<string name> : TypeDef<Test_Dialect, name> { }

// An alternate int type.
def IntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
    An alternate integer type. This type differentiates itself from the
    standard integer type by not having a SignednessSemantics parameter, just
    a width.
  }];

  let parameters = (ins "unsigned":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "int<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width);
  }];
}
```

- **Type name** : 生成的 C++ 类的名称默认为 `<classParamName>Type`（例如上例中的 `TestIntegerType`）。 这可以通过 `cppClassName` 字段覆盖。 `mnemonic` 是指定解析的asm名称。 它是可选的，不指定将意味着没有解析器或打印方法附加到此类。 
- **Type documentation**：存在`summary`和`description`字段，其使用方式与Operation中相同。 即，`summary`应该是单行的，而`description`应该是更长的解释。 
- **Type parameters**：`parameters`字段是类型参数的列表。 如果未指定任何参数（默认），则此类型被视为单例类型。 参数采用`“c++Type”:$paramName` 格式。 要将C++类型用作需要在存储构造函数中分配的参数，有两种选择： 1. 设置 `hasCustomStorageConstructor` 以生成带有刚刚声明的构造函数的 TypeStorage 类——没有定义——所以我们可以自己编写它。 2. 使用`TypeParameter` tablegen类而不是"c++Type"字符串。（后半句话我不是很懂，也还没用过。）

- **TypeParameter tablegen class** : 这用于进一步指定有关每个类型参数的属性。 它包括文档（`summary`和`syntax`）、要使用的 C++ 类型、要在存储构造函数方法中使用的自定义分配器，以及用于确定参数类型的两个实例是否相等的自定义比较器。 

```cpp
// DO NOT DO THIS!
let parameters = (ins "ArrayRef<int>":$dims);
```
默认存储构造函数盲目地按值复制字段。 它对类型一无所知。 在这种情况下，ArrayRef 需要使用 `dims = allocator.copyInto(dims)` 进行分配。 

```cpp
class ArrayRefIntParam :
    TypeParameter<"::llvm::ArrayRef<int>", "Array of ints"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

...

let parameters = (ins ArrayRefIntParam:$dims);
```
`allocator`代码块由`$_allocator`（是在其中分配对象的 TypeStorageAllocator）和`$_dst`（是放置已分配数据的变量）组成。`comparator`代码块由`$_lhs`和`$_rhs`参数类型实例组成。

自定义Type还有不少内容，但目前我没有这方面的需求，所以就没有继续看了，这里只是简单了解了一下。感兴趣的读者可以自行查看文档进行深入研究：https://mlir.llvm.org/docs/OpDefinitions/ 。

# 14. DEBUG方法
使用`mlir-tblgen`来看产生的文本。TableGen 语法有时可能很晦涩。阅读生成的文本对于理解和调试问题非常有用。 要构建 `mlir-tblgen`，可以运行 `cmake --build 。 --target mlir-tblgen` 在我们的构建目录中，并在 `bin/` 子目录中找到 `mlir-tblgen` 二进制文件。 所有支持的生成器都可以通过 `mlir-tblgen --help` 找到。 

要查看生成的代码，请通过 `-I` 提供包含路径，使用 `mlir-tblgen` 调用特定生成器。 例如：

```cpp
# To see op C++ class declaration
mlir-tblgen --gen-op-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op C++ class definition
mlir-tblgen --gen-op-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op documentation
mlir-tblgen --gen-dialect-doc -I /path/to/mlir/include /path/to/input/td/file

# To see op interface C++ class declaration
mlir-tblgen --gen-op-interface-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op interface C++ class definition
mlir-tblgen --gen-op-interface-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op interface documentation
mlir-tblgen --gen-op-interface-doc -I /path/to/mlir/include /path/to/input/td/file
```



# 15. 总结
这一节在[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 的基础上补充完整了ODS的要点。约束和属性的定义都是MLIR中相当重要的元素，至于类型的定义个人认为了解即可，等到我们需要自定义类型的时候再仔细研究。最后MLIR的语法比较晦涩，初学者可以借助`mlir-tblgen`来辅助debug。

在这两篇文章里，我跟着MLIR的ODS规范完整走了一遍并总结了14个要点，对于每一个要点我都在OneFlow MLIR的Op定义中进行了对照，并给出了一些示例代码和位置。希望对读者入门MLIR有帮助。

 