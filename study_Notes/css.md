> 最近在学css，虽然这玩意儿不必硬记，都是即用即查，但由于我学的比较emm（是打算进行一个速成来着），因此记录一些常用的知识点，以防长久不看学了白学。

### 样式定义方式

1. 行内样式表（inline style sheet）

       定义在标签的style属性中，只对该标签生效。

2. 内部样式表（internal style sheet）

       定义在style标签内，通过选择器影响对应的标签，可以对当前页面的指定多个元素生效。

3. 外部样式表（external style sheet）

       定义在css样式文件中，通过选择器影响对应的标签，可使用link标签引入各个页面，可以对多个页面产生影响。

例：

```css
<link href="/media/examples/link-element-example.css" rel="stylesheet">
```

### 选择器

1. 标签选择器

2. 类选择器 .

3. ID选择器 #

4. 伪类选择器

5. 复合选择器

6. 通配符选择器

7. 伪元素选择器

    [AcWing](https://www.acwing.com/blog/content/16243/)

### 文本

+ text-align（对齐）的选项：center(居中), left, right, justify(左右对齐)。

+ line-height（行高）可以用来竖直居中

+ letter-spaceing （字间距） 单位同行高。

+ text-indent（缩进） 单位em。

+ text-decoration 这个 CSS 属性是用于设置文本的修饰线外观的（下划线、上划线、贯穿线/删除线 或 闪烁）用到再搜。

+ text-shadow（文字阴影）: 每个阴影值由元素在**X和Y方向**的**偏移量**、**模糊半径**和**颜色值**组成。例 text-shadow: 5px -5px 2px color。

### 字体

+ font-weight：字体的粗细。

+ font-style：字体是否是倾斜的 italic(斜体)。

+ font-weight：字体的粗细程度，一些字体只提供 normal 和 bold 两种值。

+ font-family：表示的是字体，例如英文里面的带衬线字体，中文里面的草书这些。

### 背景

+ background-color：设置元素的背景色, 属性的值为颜色值或关键字”transparent”二者选其一。

+ background-image：为一个元素设置一个或者多个背景图像（渐变色：linear-gradient(rgba(0, 0, 255, 0.5), rgba(255, 255, 0, 0.5))）           url('address')

+ background-size：设置背景图片大小。图片可以保有其原有的尺寸，或者拉伸到新的尺寸，或者在保持其原有比例的同时缩放到元素的可用空间的尺寸。

+ background-repeat：定义背景图像的重复方式。背景图像可以沿着水平轴，垂直轴，两个轴重复，或者根本不重复。

+ background-position：为背景图片设置初始位置。

+ background-attachment：决定背景图像的位置是在视口内固定，或者随着包含它的区块滚动。
  
  opacity 透明度

### 边框

+ border-style：用于设定元素所有边框的样式 

+ border-width：设置盒子模型的边框宽度

+ border-color：是一个用于设置元素四个边框颜色的快捷属性，还有border-top-color$\dots$

+ border-radius：设置外边框圆角

+ border-collapse：决定表格的边框是分开的还是合并的。在分隔模式下，相邻的单元格都拥有独立的边框。在合并模式下，相邻单元格共享边框。

### 元素展示格式

+ display 
  
  - block：
    
    + 独占一行
    
    + width，height，margin，padding均可控制
    
    + width默认100%
  
  - inline：
    
    + 可以共占一行
    
    + width与height无效，水平方向的margin与padding有效，竖直反向的maigin与padding无效
    
    + width默认为本身内容高度
  
  - inline-block：
    
    + 可以共占一行
    
    + width，height，margin，padding均可控制
    
    + width默认为本身内容高度

+ white-space：用来设置如何处理元素中的 空白 (en-US)。

+ overflow：定义当一个元素的内容太大而无法适应 块级格式化上下文 时候该做什么。它是 overflow-x 和overflow-y的 简写属性。

+ text-overflow：确定如何向用户发出未显示的溢出内容信号。它可以被剪切，显示一个省略号或显示一个自定义字符串。

### 盒子模型

+ box-sizing：定义了user agent应该如何计算一个元素的总宽度和高度
  
  + content-box：是默认值，设置border和padding均会增加元素的宽高
  
  + border-box：设置border和padding不会增加宽高而是挤占内容区域

### 位置

+ position：用于指定一个元素在文档中的定位方式
  
  定位类型：
  
  + 定位元素（positioned element）是其计算后位置属性为 relative, absolute, fixed 或 sticky 的一个元素（换句话说，除static以外的任何东西）。
  
  + 相对定位元素（relatively positioned element）是计算后位置属性为 **relative** 的元素。
  
  + 绝对定位元素（absolutely positioned element）是计算后位置属性为 **absolute** 或 fixed 的元素。
  
  + 粘性定位元素（stickily positioned element）是计算后位置属性为 **sticky** 的元素。
  
  取值：
  
  + static：指定元素使用正常的布局行为，即元素在文档常规流中当前的布局位置。此时 top, right, bottom, left 和 z-index 属性无效
  
  + relative：元素先放置在未添加定位时的位置，再在不改变页面布局的前提下调整元素位置（因此会在此元素未添加定位时所在位置留下空白）。top, right, bottom, left等调整元素相对于初始位置的偏移量。
  
  + absolute：元素会被移出正常文档流，并不为元素预留空间，通过指定元素相对于最近的非 static 定位祖先元素的偏移，来确定元素位置。绝对定位的元素可以设置外边距（margins），且不会与其他边距合并。
  
  + fixed：元素会被移出正常文档流，并不为元素预留空间，而是通过指定元素相对于屏幕视口（viewport）的位置来指定元素位置。元素的位置在屏幕滚动时不会改变。
  
  + stiky：元素根据正常文档流进行定位，然后相对它的最近滚动祖先（nearest scrolling ancestor）和 containing block (最近块级祖先 nearest block-level ancestor)，包括table-related元素，基于top, right, bottom, 和 left的值进行偏移。偏移值不会影响任何其他元素的位置。

### 浮动

+ float：指定一个元素应沿其容器的左侧或右侧放置，允许文本和内联元素环绕它。该元素从网页的正常流动(文档流)中移除，尽管仍然保持部分的流动性（与绝对定位相反）。由于float意味着使用块布局，它在某些情况下修改display 值的计算值：
  
  display为inline或inline-block时，使用float后会统一变成inline-block。
  
  + 取值：
    
    - left：表示元素必须浮动在其所在的块容器左侧的关键字。
    
    - right：表明元素必须浮动在其所在的块容器右侧的关键字。

+ clear：有时，你可能想要强制元素移至任何浮动元素下方。比如说，你可能希望某个段落与浮动元素保持相邻的位置，但又希望这个段落从头开始强制独占一行。此时可以使用clear。
  
  + 取值：
    
    + left：清除左侧浮动
    
    + right：清除右侧浮动
    
    + both：清除两侧浮动

### flex布局

+ flex：设置了弹性项目如何增大或缩小以适应其弹性容器中可用的空间。
  
  + flex-direction： 属性指定了内部元素是如何在 flex 容器中布局的，定义了主轴的方向(正方向或反方向)。
    
    取值：
    
    + row：flex容器的主轴被定义为与文本方向相同。 主轴起点和主轴终点与内容方向相同。
    
    + row-reverse：表现和row相同，但是置换了主轴起点和主轴终点。
    
    + column：flex容器的主轴和块轴相同。主轴起点与主轴终点和书写模式的前后点相同。
    
    + column-reverse：表现和column相同，但是置换了主轴起点和主轴终点。
  
  + flex-wrap：指定 flex 元素单行显示还是多行显示。如果允许换行，这个属性允许你控制行的堆叠方向。
    
    取值：
    
    + nowrap：默认值，不换行。
    
    + wrap：换行，第一行在上方。
    
    + wrap-reverse：换行，第一行在下方。
  
  + flex-flow：是 flex-direction 和 flex-wrap 的简写。默认值为：row nowrap
  
  + justify-content：定义了浏览器之间，如何分配顺着弹性容器主轴(或者网格行轴) 的元素之间及其周围的空间。
    
    取值：
    
    + flex-start：默认值。左对齐。
    
    + flex-end：右对齐。
    
    + space-between：左右两段对齐。
    
    + space-around：在每行上均匀分配弹性元素。相邻元素间距离相同。每行第一个元素到行首的距离和每行最后一个元素到行尾的距离将会是相邻元素之间距离的一半。
    
    + space-evenly：flex项都沿着主轴均匀分布在指定的对齐容器中。相邻flex项之间的间距，主轴起始位置到第一个flex项的间距，主轴结束位置到最后一个flex项的间距，都完全一样。
  
  + align-items：将所有直接子节点上的align-self值设置为一个组。 align-self属性设置项目在其包含块中在交叉轴方向上的对齐方式。
    
    取值：
    
    + flex-start：元素向主轴起点对齐。
    
    + flex-end：元素向主轴终点对齐。
    
    + center：元素在侧轴居中。
    
    + stretch：弹性元素被在侧轴方向被拉伸到与容器相同的高度或宽度。stretch：弹性元素被在侧轴方向被拉伸到与容器相同的高度或宽度。
  
  + align-content：设置了浏览器如何沿着弹性盒子布局的纵轴和网格布局的主轴在内容项之间和周围分配空间。
    
    取值：
    
    + flex-start：所有行从垂直轴起点开始填充。第一行的垂直轴起点边和容器的垂直轴起点边对齐。接下来的每一行紧跟前一行。
    
    + flex-end：所有行从垂直轴末尾开始填充。最后一行的垂直轴终点和容器的垂直轴终点对齐。同时所有后续行与前一个对齐。
    
    + center：所有行朝向容器的中心填充。每行互相紧挨，相对于容器居中对齐。容器的垂直轴起点边和第一行的距离相等于容器的垂直轴终点边和最后一行的距离。
    
    + stretch：拉伸所有行来填满剩余空间。剩余空间平均地分配给每一行。
  
  + order：定义flex项目的顺序，值越小越靠前
  
  + flex-grow：设置 flex 项主尺寸 的 flex 增长系数。**负值无效，默认为 0。**
  
  + flex-shrink：指定了 flex 元素的收缩规则。flex 元素仅在默认宽度之和大于容器的时候才会发生收缩，其收缩的大小是依据 flex-shrink 的值。**负值无效，默认为1。**
  
  + flex-basis：指定了 flex 元素在主轴方向上的初始大小。
  
  + flex：flex-grow、flex-shrink、flex-basis的缩写。
    
    + 常用取值：
      
      + auto：flex: 1 1 auto
      
      + none：flex: 0 0 auto

### 响应式布局

+ media查询：当屏幕宽度满足特定条件时应用
