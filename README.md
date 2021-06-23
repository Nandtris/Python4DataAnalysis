# Python4DataAnalysis 2ed
online book refer to: https://www.bookstack.cn/read/pyda-2e-zh/11.5.md

## 4 Numpy 基础
- numpy: ndarray 多维数组对象
- 数学运算
  - data = np.random.randn(2,3)
  - data + data, data * 10
- Creat ndarraay
  - `data1 = np.array([1, 2, 3,], [4, 5, 6])`
- Method
  - shape 代表维度
  - dtype 代表类型
  - array
  - arange
  - random.randn()
  - ones
  - zeros
  - empty



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
# 分组键可以是任意适合长度的数组
states = np.array(['Shanghai', 'Shanghai', 'Beijing', 'Shanghai', 'Shanghai'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()
# 结果中没有key2列
# 这是因为df[‘key2’]不是数值数据
# 默认情况下，所有数值列都会被聚合
df.groupby('key1').mean()

# 返回分组的大小
df.groupby(['key1', 'key2']).size()
# 把数据片段组成一个字典
pieces = dict(list(df.groupby('key1')))
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
