# Python4DataAnalysis 2ed
online book refer to: https://www.bookstack.cn/read/pyda-2e-zh/11.5.md

## 4 Numpy 基础
### 4.1 numpy: ndarray 
- import numpy as np
- NumPy N维数组对象`ndarray`, 是一个快速而灵活的大数据集容器。
- 你可以利用这种数组对整块数据执行一些数学运算，其语法跟标量元素之间的运算一样。
- ndarray 其中的所有元素必须是相同类型的。
- 每个数组都有一个shape（一个表示各维度大小的元组）和一个dtype（一个用于说明数组数据类型的对象）
- 本书中“数组”、“NumPy数组”、”ndarray”时，基本上都指的是同一样东西，即ndarray对象
  ```Python
  data = np.random.randn(2,3)
  data + data, data * 10
  data.shape # (2, 3)
  data.dtype # dtype('float64')
  data.ndim  # 2 
  ```
- Creat ndarraay
  ```Python
  data2 = np.array([1, 2, 3,], [4, 5, 6])
  np.zeros(10)
  np.empty((2, 3, 2))
  np.arange(15)
  ```
- Method
  - shape 代表维度
  - dtype 代表类型
  - array
  - arange
  - random.randn()
  - ones
  - zeros
  - empty
- ndarray dtype
  ```Python
  # dtype 指定ndarray数据类型
  arr = np.array(np.random.randn(9), dtype=np.float64)
  # 转换数据类型
  int_arr = arr.astype(np.int32)
  ```
- numpy 数组运算
  - 数组使你不用编写循环即可对数据执行批量运算。
  - NumPy用户称其为矢量化（vectorization）。
  - 
  - 大小相等的数组之间的任何算术运算都会将运算应用到元素级别
  - 大小相同的数组之间的比较会生成布尔值数组
  - 数组与标量的算术运算会将标量值传播到各个元素
  - 不同大小的数组之间的运算叫做广播（broadcasting）
  
- Index and Slice
  - 一维数组
    ```Python
    arr = np.arange(10)
    # 与 list 取值类似
    arr[5]
    arr[5:8]
    # 赋值操作会广播，并且在原地修改值
    arr[5:8] = 12 
    # 切片[ : ]会给数组中的所有值赋值
    arr[5:8][:] =64
    ```
  - 二维数组
    - 各索引位置上的元素不再是标量而是一维数组
      ```Python
      arr2d = np.array([[1, 2 ,3], [4, 5, 6], [7, 8, 9]])
      arr2d[2] # array([7, 8, 9])
      arr2d[2][0] == arr2d[2, 0] # 7
      ```
  - arr3d
    - 在多维数组中，如果省略了后面的索引，则返回对象会是一个维度低一点的ndarray
    ```Python
    arr3d = np.array([[[1, 2, 3], [4, 5, 6]], 
                      [[7, 8, 9], [10, 11, 12]]])
    arr3d[0] # array([[1, 2, 3], [4, 5, 6]])
    # arr3d[1,0]可以访问索引以(1,0)开头的那些值（以一维数组的形式返回）
    arr3d[1, 0] # array([7, 8, 9]) 
    
    # 标量值和数组都可以被赋值给arr3d[0]
    old_value = arr3d[0].copy()
    arr3d[0] = 34 # array([[34, 34, 34], [34, 34, 34]])
    arr3d[0]=old_value
    ```
- 切片索引
  - arr1d
  `arr[1:6] # like Python list`
  - arrNd
    - `arr2d[:2]`它是沿着第0轴（即第一个轴）切片的。
    - 切片是沿着一个轴向选取元素的。
    - 表达式`arr2d[:2]`可以被认为是“选取arr2d的前两行”
    - 一次传入多个切片 `arr2d[:2, 1:]`,像这样进行切片时，只能得到相同维数的数组视图
    - 将整数索引和切片混合，可以得到低维度的切片 `arr2d[1, :2]` 选取第二行的前两列
    - “只有冒号”表示选取整个轴 `arr2d[:, :1]`
    - 对切片表达式的赋值操作也会被扩散到整个选区

- 布尔型索引
  - 通过布尔型索引选取数组中的数据，将总是创建数据的副本，即使返回一模一样的数组也是如此
  - Python关键字and和or在布尔型数组中无效。要使用&与|
    ```Python
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    data = np.random.randn(7, 4)
    names == 'Bob' # array([ True, False, False,  True, False, False, False])

    # 布尔型数组的长度必须跟被索引的轴长度一致
    data[names == 'Bob']
    # 以将布尔型数组跟切片、整数（或整数序列，稍后将对此进行详细讲解）混合使用
    data[names == 'Bob', 2:]

    names != "Bob"
    # 通过~对条件进行否定
    data[~(names == "Bob")]
    # 合应用多个布尔条件，使用&（和）、|（或）
    mask = (names == "Bob") | (names == "Will")
    data[mask]
    # 通过布尔型数组设置值是一种经常用到的手段
    data[data < 0] = 0
    data[names != 'Joe'] = 7
    ```
- 花式索引 Fancy indexing
  - 利用整数数组进行索引
  - 花式索引跟切片不一样，它总是将数据复制到新数组中。
- 数组转置和轴对换

### 4.2 ufuncs
- unary ufunc
  - np.sqrt(arr)
  - exp
  - 
- binary ufunc
  - add
  - maximum
- modf
  - 返回浮点数数组的小数和整数部分
  ```Python
  arr = np.random.randn(7) * 5
  remainder, whole_part = np.modf(arr)
- Ufuncs可以接受一个out可选参数，这样就能在数组原地进行操作
  ```Python
  np.sqrt(arr, arr) == arr == np.sqrt(arr) # Ture
  ```
  
### 4.3 利用数组进行数据处理
- 用数组表达式代替循环的做法，通常被称为矢量化
- 一般来说，矢量化数组运算要比等价的纯Python方式快上一两个数量级（甚至更多），尤其是各种数值计算
- 广播，是一种针对矢量化计算的强大手段

- 将条件逻辑表达为数组运算
  - np.where
    ```Python
    # np.where 三元表达式 `x if condition else y`的矢量化版本
    arr = np.random.rand(4, 4)
    # 所有正值替换为2，将所有负值替换为－2 
    np.where(arr > 0, 2, -2) 
    # 所有正值替换为2， 其余值不变
    np.where(arr > 0, 2, arr)
    ```
  
- 数学和统计方法
  - 通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算
  - sum、mean等聚合计算既可以当做数组的实例方法调用，也可以当做顶级NumPy函数使用
  - `arr.mean(), np.mean(arr)`
  - `arr.mean(axis=1)` 计算行的平均值 ？？？
  - `arr.sum(axis=0)` 计算列的的和 ？？？
  - `arr.cumsum(axis=0)` 多维数组，沿着0轴累加
  - `arr.cumprod(axis=1)` 多维数组，沿着1轴累积
- 用于布尔型数组的方法
  - 布尔值会被强制转换为1（True）和0（False）
  - sum经常被用来对布尔型数组中的True值计数
    ```Pyhton
    arr = np.random.randn(100)
    np.sum(arr > 0) # number of positive values
    ```
  - `any` 测试数组中是否存在一个或多个True
  - `all` 检查数组中所有值是否都是True
  - 这两个方法也能用于非布尔型数组，所有非0元素将会被当做True
 
- sort
  - `arr.sort(1)`就地排序则会修改数组本身, 1~轴向
  - 顶级方法np.sort返回的是数组的已排序副本

- 唯一化以及其它的集合逻辑
  ```Python
  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
  
  # 找出数组中的唯一值并返回已排序的结果
  # array(['Bob', 'Joe', 'Will'], dtype='<U4')
  np.unique(names)
  ```

### 4.4 数组文件输入输出
- np.save和np.load
- np.savez可以将多个数组保存到一个未压缩文件中
- 将数据压缩，可以使用numpy.savez_compressed
  ```Python
  arr = np.arange(10)
  np.save('some_array', arr)
  np.load('some_array.npy')

  np.savez('array_archive.npz', a=arr, b=arr)
  arch = np.load('array_archive.npz)
  arch['b']

  numpy.savez_compressed('array_archive.npz', a=arr, b=arr)
  ```
### 4.5 线性代数 点积
### 4.6 伪随机数
- 伪随机数，都是通过算法基于随机数生成器种子，在确定性的条件下生成的
- 用`np.random.seed`更改全局随机数生成种子 `np.random.seed(1234)`
- 避免全局状态，使用`numpy.random.RandomState`创建一个与其它隔离的随机数生成器
  ```Python
  rng = np.random.RandomState(1234)
  rng.randn(5)
  ```

## 5 Pandas 入门
- 数据清洗和分析工作变得更快更简单的数据结构和操作工具
- 常结合数值计算工具 NumPy和SciPy，分析库 statsmodels和scikit-learn，和数据可视化库 matplotlib 使用
- pandas 是专门为处理表格和混杂数据设计的
- NumPy 更适合处理统一的数值数组数据

### Pandas 数据结构
- Series
  - 一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成
  - `values`和`index`属性获取其数组表示形式和索引对象
  - `name`属性 `obj4.name = 'population', obj4.index.name = 'states'`
  
  - 自定义索引标签 `obj2 = pd.Series([2, 4, -5, 3], index=['b', 'a', 'c', 'd'])`
  - 就地修改索引 `obj2.index=['Bob', 'Steve', 'Jeff', 'Rayn']`
  - 取值赋值：以索引的方式选取Series中的单个或一组值 `obj2[['a', 'b', 'c']]= 9 `
  
  - 自动对齐：运算中根据索引标签自动对齐数据进行运算 `obj3 + obj2`
  - 运算：使用NumPy函数或类似NumPy的运算（如根据布尔型数组进行过滤、标量乘法、应用数学函数等）都会保留索引值的链接
    ```Python
    obj2[obj2>5]
    obj2 * 2
    np.exp(obj2)
    ```
  - 可以将Series看成是一个定长的有序字典 `"b" in obj2 # True`
  - 通过字典来创建Series
    ```Python
    # 只传入一个字典，则结果Series中的索引就是原字典的键（有序排列）
    sdata = {'Ohio': 35000, 'Texas': 71000, "Oregen":16000, 'Otah':5000}
    obj3 = pd.Series(sdata)
    
    # 可以传入排好序的字典的键以改变顺序
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    obj4 = pd.Series(sdata, index=states)    
    ```
  - `isnull`和`notnull`函数可用于检测缺失数据 `obj4.isnull(), pd.notnull(obj4)`

- DataFrame
  - DataFrame是一个表格型的数据结构
  - DataFrame既有行索引也有列索引
  - DataFrame中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）
  - 建DataFrame,方法之一直接传入一个由等长列表或NumPy数组组成的字典
    ```Python
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002, 2003],
            'pop': [1.8, 1.3, 1.3, 1.9, 1.4, 1.7]}
    frame = pd.DataFrame(data) # 按字典给出顺序排列DataFrame
    
    # 指定列顺序
    pd.DataFrame(data, columns=['pop', 'year', 'state'])
    
    # 缺失值 data中找不到数据的列，结果中填入缺失值 NaN
    frame2 = pd.DataFrame(data, columns=['state', 'year', 'pop', 'debt'], 
                          index = ['one', 'two', 'three', 'four', 'five', 'six'])
    
    # 字典标记方式取值，DataFrame的列获取为一个Series, name属性也设置好了
    frame2['state']
    frame2['state'].name # 'state'
    
    # frame2[column]适用于任何列的名，
    # frame2.column 属性名形式只有在列名是一个合理的Python变量名时才适用
    frame2.state
    frame2['state']
    
    # 列可以通过赋值的方式进行修改
    frame2['debt'] = 17
    # 将列表或数组赋值给某个列时，其长度必须跟DataFrame的长度相匹配
    frame2['debt'] = np.arange(6)
    # 如果赋值的是一个Series，就会精确匹配DataFrame的索引，所有的空位都将被填上缺失值
    val = pd.Series([-1.2, -1.5, -3.5], index=['two', 'four', 'six'])
    frame2['debt']=val
    
    # 行也可以通过位置或名称的方式进行获取
    frame2.loc['one']
    
    # 为不存在的列赋值会创建出一个新列
    # 不能用frame2.eastern创建新的列
    # 添加一个新的布尔值的列，state是否为’Ohio’, 是为True，否填充值False
    frame2['estern'] = frame2.state == 'Ohio' 
    
    # 关键字del用于删除列
    del frame2['estern']
    
    # 嵌套字典传给DataFrame，
    # pandas就会被解释为：外层字典的键作为列，内层键则作为行索引
    dic = {'Beijing':{2001: 23, 2002: 35, 2003:56},
           'Hangzhou':{2002: 2, 2003: 3, 2004: 5}}
    frame3 = pd.DataFrame(dic)
    
    # name 属性
    # values 属性以二维ndarray的形式返回DataFrame中的数据
    frame3.index.name = 'year'
    frame3.columns.name = 'cty'
    frame2.values
    ```

- 索引对象
  - pandas的索引对象负责管理轴标签和其他元数据（比如轴名称等）。
  - 构建Series或DataFrame时，所用到的任何数组或其他序列的标签都会被转换成一个Index
  - Index对象是不可变的，因此用户不能对其进行修改

### 5.2 基本功能
- 操作Series和DataFrame中的数据的基本手段
- Reindex
  - pandas对象的方法，创建一个新对象，它的数据符合新的索引
  - Series
    ```Python
    obj = pd.Series([1, 2, 3], index=['b', 'a', 'c'])
    # reindex将会根据新索引进行重排。
    # 如果某个索引值当前不存在，就引入缺失值
    obj.reindex(['a', 'b', 'c', 'd'])
    
    # 像时间序列这样的有序数据
    # 重新索引时插值处理
    obj3 = pd.Series(['bluer', 'oranger', 'reder'], index=[0, 2, 5])
    obj3.reindex(range(6), method='ffill')
    ```
  - DataFrame 类似Series
    ```Python
    frame = pd.DataFrame(np.arange(9).reshape(3, 3),
                         index = ['a', 'c', 'b'],
                         columns = ['Ohio', 'Texas', 'California'])
    frame2 = frame.reindex(['a', 'b', 'c', 'd'])
    frame3 = frame.reindex(['a', 'b', 'c', 'd'], 
                          columns=['Hangzhou', 'Texas', 'California'])
    ```

- drop
  - 丢弃某条轴上的一个或多个项，只要有一个索引数组或列表即可
  - drop方法返回的是一个删除了指定轴值后新对象
  - 如果就地修改，传入参数 `inplace=True`
    ```Python
    # Series
    obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
    new_obj = obj.drop('c')
    new_obj = obj.drop(['c', 'd'])

    # DataFrame，可以删除任意轴上的索引值
    data = pd.DataFrame(np.arange(16).reshape(4, 4),
                       index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                       columns = ['one', 'two', 'three', 'four'])
    # 用标签序列调用drop会从行标签（axis 0）删除值
    data.drop(['Ohio', 'Utah'])
    # 通过传递axis=1或axis=’columns’可以删除列的值
    data.drop(['one', 'two'], axis=1)
    # 就地修改对象，不会返回新的对象
    data.drop(['one', 'two'], axis=1, inplace=True)
    ```
- 索引 选取 过滤
  ```Python
  # Series
  obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
  obj['b']
  obj[2:4]
  obj[['b', 'a', 'd']]
  obj[[1, 3]]
  obj[obj < 2]
  
  # DataFrame
  data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
  # 用一个值或序列对DataFrame进行索引其实就是获取一个或多个列
  向[ ]传递单一的元素或列表，就可选择列
  data['two']
  data[['three', 'one']]
  
  #切片或布尔型数组选取数据
  data[:2] # 前两行
  data[data['three'] > 5]
  
  data < 5
  data[data < 5] = 0
  ```
- 轴标签（loc）或整数索引（iloc）
  
  ```Python
  data.loc['Colorado', ['two', 'three']]
  data.iloc[2, [3, 0, 1]]
  data.iloc[[1, 2], [3, 0, 1]]
  
  data.loc[:'Utah', 'two']
  data.iloc[:, :3][data.three > 5]
  ```
  
- 整数索引
  - 如果轴索引含有整数，数据选取总会使用标签。
  - 为了更准确，请使用loc（标签）或iloc（整数）
  
- 算术运算和数据对齐
  - 对不同索引的对象进行算术运算
  - 在将对象相加时，如果存在不同的索引对，则结果的索引就是该索引对的并集
  - 加法相当于SQL中的join， 非共有部分引入缺失值 NaN, 共有部分值加总
    - 自动的数据对齐操作在不重叠的索引处引入了NA值
    - 对于DataFrame，对齐操作会同时发生在行和列上
    - DataFrame对象相加，没有共用的列或行标签，结果都会是空
  - add, sub, div, floordiv, mul, pow
  - radd, rsub, rdiv, rfloordiv, rmul, rpow
  - df1.rdiv(1) = 1/df1
    ```Python
    df1 = pd.DataFrame(np.arange(12).reshape((3, 4)),
                  columns = list('abcd'))
    df2 = pd.DataFrame(np.arange(20).reshape((4, 5)),
                      columns = list('abcde'))
    df2.loc[1, 'b'] = np.nan
    
    # 将它们相加时，没有重叠的位置就会产生NA值
    df1 + df2
    
    # 对不同索引的对象进行算术运算时，
    # 当一个对象中某个轴标签在另一个对象中找不到时填充一个特殊值（比如0）
    df1.add(df2, fill_value=0)
    
    1/df1 = df1.rdiv(1)
    ```
- DataFrame和Series之间算术运算
  - arr2d减去arr2d[0]，每一行都会执行这个操作。这就叫做广播（broadcasting）
  ```Python
  frame = pd.DataFrame(np.arange(12.).reshape(4, 3),
                    columns = list('bde'),
                    index = ['Utah', 'Ohio', 'Texas', 'Oregen'])
  series = frame.iloc[0]
  # 默认情况下，算术运算会将Series的索引匹配到DataFrame的列，然后沿着行一直向下广播
  frame - series

  # 如果某个索引值在DataFrame的列或Series的索引中找不到，
  # 则参与运算的两个对象就会被重新索引以形成并集
  series2 = pd.Series(range(3), index=['b', 'e', 'f'])
  frame + series2

  # 匹配行且在列上广播，则必须使用算术运算方法
  series3 = frame['d']
  frame.sub(series3, axis='index')
  ```
### 5.3 汇总和计算描述统计
- 从Series中提取单个值（如sum或mean）
- 从DataFrame的行或列中提取一个Series。
- 跟对应的NumPy数组方法相比，它们都是基于没有缺失数据的假设而构建的
- count describe min max argmin argmax idxmin idxmax quantile
- sum mean median mad std var skew kurt 
- cumsum cumprod cummin cummax diff pct_cahnge
  ```Python
  df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [.75, -1.3]],
            columns = ['one', 'two'],
            index = ['a', 'b', 'c', 'd'])
            
  # 调用DataFrame的sum方法将会返回一个含有列的和的Serie
  df.sum()
  
  # 传入axis=’columns’或axis=1将会按行进行求和运算
  df.sum(axis=1)
  
  # NA值会自动被排除,skipna选项可以禁用该功能
  df.mean(axis='columns', skipna=Fasle)
  
  # 一次性产生多个汇总统计
  df.describe()
  
  df:
    one 	two
  a 	1.40 	NaN
  b 	7.10 	-4.5
  c 	NaN 	NaN
  d 	0.75 	-1.3
  
  # 计算百分数变化
  df.pct_change() # (7.1-1.4)/1.4 = 4.071429
      one 	two
  a 	NaN 	NaN
  b 	4.071429 	NaN
  c 	0.000000 	0.000000
  d 	-0.894366 	-0.711111
  ```
  
- 相关系数与协方差 ???

- 唯一值、值计数以及成员资格
  - 从一维Series的值中抽取信息
    ```Python
    obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
    
    obj.unique()
    obj.value_counts()
    pd.value_counts(obj.values, sort=False)
    
    # isin用于判断矢量化集合的成员资格
    mask = obj.isin(['b', 'c'])
    obj[mask]
    ```



## 10 数据聚合与分组

将数据集加载、融合、准备好之后，通常就是计算分组统计或生成透视表。<br>
pandas提供了一个灵活高效的gruopby功能，它使你能以一种自然的方式对数据集进行切片、切块、摘要等操作。<br>
- 使用一个或多个键（形式可以是函数、数组或DataFrame列名）分割pandas对象;
- 计算分组的概述统计，比如数量、平均值或标准差，或是用户定义的函数;
- 应用组内转换或其他运算，如规格化、线性回归、排名或选取子集等;
- 计算透视表或交叉表;
- 执行分位数分析以及其它统计分组分析
- 对时间序列数据的聚合(ch11)

### 10.1 聚合与分组运算
分组键可以有多种形式，且类型不必相同：<br>
- 列表或数组，其长度与待分组的轴一样
- 字典或Series，给出待分组轴上的值与分组名之间的对应关系
- 函数，用于处理轴索引或索引中的各个标签
- Groupby
```Python
df = pd.DataFrame({'data1': np.random.randn(5),
                  'data2': np.random.randn(5),
                  'key1': ['a', 'a', 'b','b','a'],
                  'key2': ['one', 'one', 'two', 'one', 'one']})
# 变量grouped是一个GroupBy对象
# 含有一些有关分组键df[‘key1’] df['key2'] 的中间数据
grouped = df['data1'].groupby(df['key1'], df['key2'])
# groupby 对象支持迭代
# 产生一组二元元组（由分组名和数据块组成）
for name, group in grouped:
    print(name)
    print(group)


# Series 分组键
grouped = df['data1'].groupby(df['key1'], df['key2']).mean()

# 结果中没有key2列
# 这是因为df[‘key2’]不是数值数据
# 默认情况下，所有数值列都会被聚合
df.groupby('key1').mean()

# 返回分组的大小
df.groupby(['key1', 'key2']).size()
# 把数据片段组成一个字典
pieces = dict(list(df.groupby('key1')))
```
- 选取列、列地子集
```Python
df.groupby('key1')['data1'].mean() # Series
# ≈
df[['data1']].groupby(df['key1']).mean() # DataFrame
```

- 分组依据
  - 数组
    ```Python
    # 分组键可以是任意适合长度的数组
    states = np.array(['Shanghai', 'Shanghai', 'Beijing', 'Shanghai', 'Shanghai'])
    years = np.array([2005, 2005, 2006, 2005, 2006])
    df['data1'].groupby([states, years]).mean()
    ```
  - 字典或Series
    ```Python
    # dict & Series
    people = pd.DataFrame(np.random.randn(5, 5),
                          columns = ['a', 'b', 'c', 'd', 'e'],
                          index = ['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
    people.iloc[2:3, [1, 2]] = np.nan
    mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'e': 'red', 'f': 'orange'}
    by_column = people.groupby(mapping, axis=1).sum()
    by_column
    
    # Series
    map_series = pd.Series(mapping)
    people.groupby(map_series, axis=1).count()
    ```
    
  - 函数
    ```Pyrthon
    key_list = ['one', 'one', 'one', 'two', 'two']
    people.groupby([len, key_list]).min()
    ```
    
  - 索引级别
  
    ```Python
    # 层次化索引
    columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]],
                                    names = ['cty', 'tenor'])
    hier_df = pd.DataFrame(np.random.randn(4, 5), columns = columns)
    # 要根据级别分组，使用level关键字传递级别序号或名字
    hier_df.groupby(level='cty', axis=1).count()
    ```

### 10.2 数据聚合
- GroupBy 
  - 自定义聚合函数，性能开销大
    - aggregate/agg
    ```Python
    def peak_to_peak(arr):
        return arr.max() - arr.min()
    df.groupby('key1').agg(peak_to_peak)
    ```
- 面向列的多函数应用
  ```Python
  tips = pd.read_csv('examples/tips.csv')
  tips['tip_pct'] = tips['tip'] / tips['total_bill']
  
  # agg 自定义函数
  grouped = tips.groupby(['day', 'smoker'])
  grouped_pct = grouped['tip_pct']
  grouped_pct.agg('mean')
  
  # 一列应用一组聚合函数
  grouped_pct.agg(['mean', 'std', peak_to_peak])
  # 以元组形式自定义聚合后列名
  grouped_pct.agg([('foo', 'mean'), ('bar', np.std)]) # foo 替换 mean 列名
  
  # DataFrame 类似以上
  functions = ['count', 'mean', 'max']
  result = grouped['tip_pct', 'total_bill'].agg(fuctions) # 形成层次化列
  # 自定义列名
  ftuples = [('feng', 'mean'), ('jing', np.var)]
  grouped['tip_pct', 'total_bill').agg(ftuples)
  
  # 不同列应用不同的函数
  grouped.agg({'tip': np.max, 'size': 'sum'})
  # 同列应用不同函数
  grouped.agg({'tip_pct': ['min', 'max', 'mean', 'std'],
               'size': 'sum'})
  ```
- 以“没有行索引”的形式返回聚合数据
  
- 优化后的聚合函数
  - min max std var sum mean prod last first count median

### 10.3 Apply
- agg apply区别？？？
  ```Python
  def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]
  
  tips = pd.read_csv('examples/tips.csv')
  tips['tip_pct'] = tips['tip'] / tips['total_bill']
  
  tips.groupby('smoker').apply(top)
  tips.groupby(['smoker','day']).apply(top, n=1, column='total_bill')
  ```
- 禁止分组
  `tips.groupby('smoker', group_keys=False).apply(top)`
- 分位数
  ```Python
  frame = pd.DataFrame({'data1': np.random.randn(1000), 'data2': np.random.randn(1000)})
  
  quartiles = pd.cut(frame.data1, 4)
  def get_statas(group):
      return {'min': group.min(), 'max': group.max(), 
             'count': group.count(), 'mean': group.mean()}
             
  frame.data2.groupby(quartiles).apply(get_statas).unstack()
  
  grouping = pd.qcut(frame.data1, 10, labels=False)
  grouped = frame.data2.groupby(grouping)
  grouped.apply(get_statas)
  ```
### 10.4 pivot_table cross_tabulation
- picot_table
```pd.pivot_table(data, values=None, index=None, columns=None, 
                  aggfunc='mean', fill_value=None, margins=False, 
                  dropna=True, margins_name='All')
```
- example
```Python
tips.pivot_table(['tip', 'total_bill'], index=['time', 'day'], 
                 columns='smoker', margins=True)
                 
tips.pivot_table('tip_pct', index=['time', 'smoker'], columns='day',
                 aggfunc=len, margins=True)
                 
tips.pivot_table('tip_pct', index=['time', 'size', 'smoker'], 
                 columns='day', aggfunc='mean', margins=True, fill_value=0)
```

## 11 时间序列
对时间序列数据的聚合（groupby的特殊用法之一）也称作重采样（resampling）<br>
- time Series
  - timestamp  特定的时刻
  - periods    如2007年1月或2010年全年
  - interval   由起始和结束时间戳表示
- pandas提供了许多内置的时间序列处理工具和数据算法。
- 可以高效处理非常大的时间序列，轻松地进行切片/切块、聚合、对定期/不定期的时间序列进行重采样等。
- 有些工具特别适合金融和经济应用，你当然也可以用它们来分析服务器日志数据。

### 11.1 
- from datetime import datetime
- date     以公历形式储存日历日期（年月日）
- time     时间储存为时分秒毫秒
- datetime 以毫秒形式存储日期和时间
- delta    两个datetime 值之间的差（日，秒， 毫秒）
- str <==> datetime
  - str
  - datetime.strftime("%Y-%m-%d")
  - datetime.strptime(strvalue, "%Y-%m-%d")

### 11.2
- 切片
  - 可以传入字符串日期、datetime或Timestamp 对pd.Series/DateFrame时间序列进行切片
  - 切片所产生的是原时间序列的视图，跟NumPy数组的切片运算是一样的；
  - 这意味着，没有数据被复制，对切片进行修改会反映到原始数据上
  
- 带有重复索引的时间序列
  - 对这个时间序列进行索引，要么产生标量值，要么产生切片
  - 对具有非唯一时间戳的数据进行聚合
    - 用groupby，并传入level=0 
    - pd.Series.groupby(level=0).mean()

### 11.3
- 生成日期范围
  - pandas.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)
  - 默认情况下，date_range会产生按天计算的时间点
  - 默认会保留起始和结束时间戳的时间信息
  - 产生一组被规范化（normalize=True）到午夜的时间戳
  - pandas.date_range('2012-04-01', '2012-06-01')
- 频率和日期偏移量
  - pd.date_range('2000-01-01', periods=10, freq='1h30min')
  - 传入频率字符串（如”1h30min”），这种字符串可以被高效地解析为等效的表达式
  - 11-4 时间序列的基础频率
- 用的频率类 WOM（Week Of Month）
  - pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI')
  - 每月第三个星期五
- 移动（超前和滞后）数据
  - 移动（shifting）指的是沿着时间轴将数据前移或后移
  ``` Python
  ts = pandas.Series()
  # 对数据位移
  ts.shift(2)
  ts.shift(-2)
  # 对时间戳位移
  ts.shift(2, freq='M')
  ts.shift(1, freq='90T')
  ```
  - shift通常用于计算一个时间序列或多个时间序列（如DataFrame的列）中的百分比变化
    - `ts / ts.shift(1) - 1` ?
- 通过偏移量对日期进行位移
  - pandas的日期偏移量还可以用在datetime或Timestamp对象上
  ``` Python
  from pandas.tseries.offsets import MonthEnd, Day
  offset = MonthEnd()
  ts = pd.Series(np.random.randn(20), 
                 index = pd.date_range('1/15/2000', periods=20, freq='4d')
  ts.groupby(offset.rollforward).mean()
  ts.resample('M').mean() # 效果同上
  ```
### 11.4 Time Zone
- pd.Series.tz_localize('UTC')
- pd.Series.tz_converse('Eroupe/Berlin')
- pandas 时间序列默认为单纯时间(native)，时区为 None,
- 将时间本地化(localize)后，就可以转变(converse)其他时区
- Timestamp
  - `stamp_utc = pd.Timestamp('2011-03-12 04:00', tz='UTC').tz_convert('Asia/Shanghai')`
  - `stamp_NY = pd.Timestamp('2011-03-12 04:00').tz_localize('UTC').tz_convert('America/New_York')`
  - Timestamp对象在内部保存了一个UTC时间戳值（自UNIX纪元（1970年1月1日）算起的纳秒数）,
  - 这个UTC值在时区转换过程中是不会发生变化的
    - `stamp_utc.value == stamp_NY # True`
  - 如果两个时间序列的时区不同，在将它们合并到一起时，最终结果就会是UTC,
  - 由于时间戳其实是以UTC存储的，所以这是一个很简单的运算，并不需要发生任何转换


### 11.5 时期及其算术运算
- periods
  - `p = pd.period(2017, 'A-DEC') # Period('2007', 'A-DEC')`
  - 从2007年1月1日到2007年12月31日之间的整段时间
  -  `p + 5 # Period('2022', 'A-DEC')`
  -  period_range
    - `pd.period_range('2006-12-03', '2021-12-09', freq='M') # 产生 periodindex 类`
    - PeriodIndex类保存了一组Period，它可以在任何pandas数据结构中被用作轴索引
- 频率转换
  - `pd.Period(2017, freq='A-DEC').asfreq('M', how='start') # Period('2007-01', 'M')'
- 按季度计算的时期频率 Q-JAN到Q-DEC
  - 时间戳与时期转换：to_timestamp() to_period()
  ``` Python
  p = pd.Period('2019Q4', freq='Q-JAN')  # Period('2012Q4', 'Q-JAN')
  p.asfreq('D', 'start')  # Period('2011-11-01', 'D')
  # 该季度倒数第二个工作日下午4点的时间戳
  # Period('2012-01-30 16:00', 'T')
  p4pm = (p.asfreq('B', 'e')-1).asfreq('T', 's') + 16*60 # e~end, s~start
  p4pm.to_timestamp() # Timestamp('2014-01-30 16:00:00')
  ```
- 通过数组创建PeriodIndex(合并大数据集中的分开的时间列)
  ```Python
  data = pd.read_csv()
  index = data.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
  data.index = index
  data.infl
  ```
### 11.6  重采样及频率转换
- esample 各种频率转换工作的主力函数。
- resample有一个类似于groupby的API，调用resample可以分组数据，然后可调用一个聚合函数
  ```Python
  rng = pd.date_range('2000-01-01', periods=100, freq='D')
  ts = pd.Series(np.random.randn(len(rng)), index=rng)
  ts.resample('M').mean()
  ts.resample(;M', kind='period').mean()
  ```
- 降采样
  ```Python
  rng = pd.date_range('2000-01-01', periods=12, freq='T')
  ts = pd.Series(np.arange(len(rng)), index=rng)
  # 以左边界index 为 label，计数数据包含左边界，不包含右边界
  ts.resample("5min").sum() 
  # 5min内，包含右边界，以右边界为label,但偏移-1秒
  ts.resample('5min', closed='right', label='right', loffset='-1s').sum() 
  ```
- 升采样和插值
  ```Python
  frame = pd.DataFrame(np.random.randn(2, 4),
                     index = pd.date_range('01/01/2000', periods=2, freq='W-WED'),
                     columns = ['Beijing', 'Shanghai', 'Guangzhou', 'Hangzhou'])
  # 以天为周期插值采样
  df_daily = frame.resample('D').asfreq()
  frame.resample('D').ffill(limit=2) # 从前面采样填充值
  frame.resample('W-THU').ffill()
  ```
### 11.7 moving window function
  ```python
  DataFrame.rolling(window, min_periods=None, center=False, win_type=None, 
                    on=None, axis=0, closed=None)
  # 链接：https://www.jianshu.com/p/b8c795345e93

  rolling(5, min_periods=2).mean()
  # 处理缺失值问题
  # 最小观测值 min_periods 参数 从最小 2 开始计算累计均值，一直到 5，
  # 而后再开始按照每个窗口5个值依次滑动

  01 None
  02 mean = (02 + 03)/ 2 = 7.425
  03 mean = (02 + 03 + 06) / 3 = 7.433333
  04 mean = (02 + 03 + 06 + 07) / 4 = 7.432500

  DataFrame.expanding(min_periods = 1，center = False，axis = 0)
  # rolling()函数，是固定窗口大小，进行滑动计算，
  # expanding()函数只设置最小的观测值数量，不固定窗口大小，实现累计计算，即不断扩展；
  # expanding()函数，类似cumsum()函数的累计求和，其优势在于还可以进行更多的聚类计算；
  ```

## 12 Advanced pandas
### 12.1 分类数据
- pandas的分类类型，可提高性能和内存的使用率
- 统计和机器学习中使用分类数据的工具
-
- 用整数表示的方法称为分类或字典编码表示法
- 不同值得数组称为分类、字典或数据级
- 表示分类的整数值称为分类编码或简单地称为编码
  ```Python
  values = pd.Series(['0', '1', '0', '0']*2)
  dim = pd.Series(['apple', 'orange'])
  
  pd.unique(values)
  pd.value_counts(values)
  
  # take方法存储原始的字符串Series
  dim.take(values)
  ```
 - Categorical Type in pandas
  - `Categories` 分类类型，用于保存使用整数分类表示法的数据
    ```Python
    fruits = ['apple', 'orange', 'apple', 'apple']*2
    N = len(fruits)
    df = pd.DataFrame({'fruit': fruits,
                      'basket_id': np.arange(N),
                      'count': np.random.randint(3, 15, size=N),
                      'weight': np.random.uniform(0, 4, size=N)},
                      columns = ['basket_id', 'fruit', 'count', 'weight'])
    # 转变为分类对象
    # fruit_cat的值不是NumPy数组，而是一个pandas.Categorical实例
    fruit_cat = df['fruit'].astype('category')

    c = fruit_cat.values
    type(c) # pandas.core.arrays.categorical.Categorical

    # 分类对象有categories和codes属性
    c.categories # Index(['apple', 'orange'], dtype='object')
    c.codes # array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int8)

    # 从其它Python序列直接创建pandas.Categorical
    pd.Categorical(['foo', 'bar', 'baz','foo', 'bar'])

    # 使用from_codes构造器
    categories = ['bar', 'foo', 'baz']
    codes = [0, 1, 2, 0, 0, 1]
    my_cats_2 = pd.Categorical.from_codes(codes, categories)

    # 分类变换不认定指定的分类顺序
    # 使用from_codes或其它的构造器时，可以指定分类一个有意义的顺序
    ordered_cat = pd.Categorical.from_codes(codes, categories, ordered=True)

    无序的分类实例可以通过as_ordered排序
    my_cats_2.as_ordered() # result = ordered_cat
    ```
- 用分类进行计算
  ```Python
  np.random.seed(12345)
  draws = np.random.randn(1000)
  # 等长 cut
  bins = pd.cut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
  bins.codes[:10]
  bins = pd.Series(bins, name='quantile')
  result = pd.DataFrame(pd.Series(draws)
                       .groupby(bins)
                       .agg(['count', 'mean', 'max', 'min'])
                       .reset_index())
  result['quantile']
  ```
  
- 用分类提高性能
- 分类方法
  ```Python
  s = pd.Series(['a', 'b', 'c', 'd'] * 2)
  cat_s = s.astype('category')
  # cat属性提供了分类方法的入口
  cat_s.cat.codes
  cat_s.cat.categories # Index(['a', 'b', 'c', 'd'], dtype='object')

  # 假设这个数据的实际分类集，超出了数据中的四个值
  # 我们可以使用set_categories方法改变它们
  actual_category = ['a', 'b', 'c', 'd', 'e']
  cat_s2 = cat_s.cat.set_categories(actual_category)

  cat_s.value_counts()
  cat_s.value_counts()

  # 在大数据集中，分类经常作为节省内存和高性能的便捷工具
  # 过滤完大DataFrame或Series之后，许多分类可能不会出现在数据中。
  # 用remove_unused_categories方法删除没看到的分类
  cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
  cat_s3.cat.remove_unused_categories()
  ```
  

- 为建模创建虚拟变量

### 12.2 GroupBy高级应用 ???
- 12.1 分组转换和“解封”GroupBy
- 12.2 分组的时间重采样
- 12.3 链式编程技术
   ```Python
   result = (load_data()
            [lambda x: x.col2 < 0]
            .assign(col1_demeaned=lambda x: x.col1 - x.col1.mean())
            .groupby('key')
            .col1_demeaned.std())
   ```
   
## 14 case study
### 14.1 Bitly的USA.gov数据
- 该数据集中最常出现的是哪个时区
- 按Windows和非Windows用户对时区统计信息进行分解
- seaborn 可视化

  ```Python
  import numpy as np
  import pandas as pd
  from pandas import Series, DataFrame

  import json
  
  path = 'datasets/bitly_usagov/example.txt'
  records = [json.loads(line) for line in open(path)]
  # records[:2]
  # 并不是所有记录都有时区字段
  time_zones  = [rec['tz'] for rec in records if 'tz' in rec]
  # time_zones[:11]

  # 该数据集中最常出现的是哪个时区（即tz字段）
  # 1 对时区进行计数:只使用标准Python库
  def get_counts(sequence):
      counts = {}
      for x in sequence:
          if x in counts:
              counts[x] += 1
          else:
              counts[x] = 1
      return counts
  counts = get_counts(time_zones)
  # counts['America/New_York']
  # 得到前10位的时区及其计数值
  def top_counts(count_dict, n=10):
      value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
      value_key_pairs.sort() # sort()原地排序， 不返回值
      return value_key_pairs[-n:]    
  top_counts(counts)    

  # 2 对时区进行计数:用pandas对时区进行计数
  frame = pd.DataFrame(records)
  # frame.info() # 摘要视图（summary view）
  # frame.tz[:10]
  tz_counts = frame['tz'].value_counts()
  tz_counts


  # 可视化这个数据
  # fillna替换缺失值 NA，
  # 而未知值（空字符串）则可以通过布尔型数组索引加以替换
  clean_tz = frame['tz'].fillna('Missing')
  clean_tz[clean_tz == ""] = "Unkonwn"
  tz_counts = clean_tz.value_counts()
  tz_counts["Missing"] # 120

  import seaborn as sns
  # subset = tz_counts[:10]
  # sns.barplot(y=subset.index, x=subset.values)



  # frame a~agent 字段含有执行URL短缩操作的浏览器、设备、应用程序的相关信息 
  # 解析 agent
  # 将这种字符串的第一节（与浏览器大致对应）分离出来
  # 并得到另外一份用户行为摘要
  results  = pd.Series([x.split()[0] for x in frame.a.dropna()])
  results.value_counts()[:10]


  # 按Windows和非Windows用户对时区统计信息进行分解
  # 只要agent字符串中含有”Windows”就认为该用户为Windows用户
  # 有的agent缺失，所以首先将它们从数据中移除
  cframe = frame[frame.a.notnull()]
  # 计算出各行是否含有Windows的值
  cframe['os'] = np.where(cframe['a'].str.contains('Windows'),
                            'Windows', 'Not Windows')
  # cframe['os'][:5]
  by_tz_os = cframe.groupby(['tz', 'os'])
  # https://blog.csdn.net/MissingDi/article/details/106982381
  # groupby.size()/count() 区别：size计数分类项本身
  agg_counts = by_tz_os.size().unstack().fillna(0)
  # 选取最常出现的时区
  # 根据agg_counts中的行数构造了一个间接索引数组
  # Use to sort in ascending order
  # 按小到大顺序排列`agg_counts.sum(1)`，然后取其索引
  indexer = agg_counts.sum(1).argsort()
  # 截取最后10行最大值: 按时区、浏览器分类计数
  count_subset = agg_counts.take(indexer[-10:])

  # 传递一个额外参数到seaborn的barpolt函数，来画一个堆积条形图
  count_subset = count_subset.stack()
  count_subset.name ='total'
  count_subset = count_subset.reset_index()
  # sns.barplot(x='total', y='tz', hue='os',  data=count_subset)

  # 上面图不容易看出Windows用户在小分组中的相对比例，
  # 因此标准化分组百分比之和为1
  # 以 'tz' 分组，每个组内 Win + NotWin = 1
  def norm_total(group):
      group['normed_total'] = group.total / group.total.sum()
      return group
  results = count_subset.groupby('tz').apply(norm_total)
  sns.barplot(x='normed_total', y='tz', hue='os',  data=results)
  ```
  
### 12.2 MovieLens 1M dataset
GroupLens Research（http://www.grouplens.org/node/73 ）<br>
- 含有来自6000名用户对4000部电影的100万条评分数据
- 根据性别和年龄计算某部电影的平均得分
- 了解女性观众最喜欢的电影
- 找出分歧最大的电影（不考虑性别因素），std
  ```Python
  import pandas as pd
  pd.options.display.max_rows = 12

  unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
  users = pd.read_table('datasets/movielens/users.dat', sep = '::',
                        header = None, names = unames)
  rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_table('datasets/movielens/ratings.dat', sep = '::',
                          header = None, names = rnames)
  mnames = ['movie_id', 'title', 'genres']
  movies = pd.read_table('datasets/movielens/movies.dat', sep = '::', 
                         header = None, names = mnames)

  # 根据性别和年龄计算某部电影的平均得分
  data = pd.merge(pd.merge(ratings, users), movies)
  # 按性别计算每部电影的平均得分，可以使用pivot_table方法
  mean_ratings = data.pivot_table('rating', index = 'title', 
                                  columns = 'gender', aggfunc = 'mean')
  # mean_ratings[:5]

  # 过滤掉评分数据不够250条的电影
  # groupby().size()得到一个含有各电影分组大小的Series对象
  ratings_by_title = data.groupby('title').size()
  active_titles = ratings_by_title.index[ratings_by_title > 250]
  mean_ratings = mean_ratings.loc[active_titles]

  # 了解女性观众最喜欢的电影，我们可以对F列降序排列
  top_female_ratings = mean_ratings.sort_values(by = 'F', ascending = False)
  # top_female_ratings[:10]

  # 男性和女性观众分歧最大的电影
  # 给mean_ratings加上一个用于存放平均得分之差的列
  mean_ratings['diff'] = mean_ratings.M - mean_ratings.F
  # 按”diff”排序即可得到分歧最大且女性观众更喜欢的电影
  sort_by_diff = mean_ratings.sort_values('diff')
  # 对排序结果反序并取出前10行，得到的则是男性观众更喜欢的电影
  sort_by_diff[::-1][:10]

  # 只是想要找出分歧最大的电影（不考虑性别因素），
  # 则可以计算得分数据的方差或标准差
  rating_std_by_title = data.groupby('title')['rating'].std()
  rating_std_by_title = rating_std_by_title.loc[active_titles]
  rating_std_by_title.sort_values( ascending = False)[:10]
  ```
### 12.3 
