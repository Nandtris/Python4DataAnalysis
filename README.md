# Python4DataAnalysis 2ed

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

## 11 时间序列

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
-

## 12 Advanced pandas
### 1 Categorical Type in pandas
- 
