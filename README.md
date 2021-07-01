# Python4DataAnalysis 2ed
online book refer to: https://www.bookstack.cn/read/pyda-2e-zh/11.5.md

## 4 Numpy 基础
### 4.1 numpy: ndarray 
- import numpy as np
- NumPy N维数组对象（即ndarray, 是一个快速而灵活的大数据集容器。
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
### 1 Categorical Type in pandas
- 
