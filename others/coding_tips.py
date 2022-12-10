# 1.编写抽象类，方便不同使用者实现不同的功能
import os
from abc import ABCMeta, abstractmethod

"""Base processor to be used for all preparation."""
class DataProcessor(metaclass=ABCMeta):
  def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory
  @abstractmethod
  def read(self):
   """Read raw data."""
  @abstractmethod
  def process(self):
   """Processes raw data. This step should create the raw dataframe with all the required features. Shouldn't implement statistical or text cleaning."""
  @abstractmethod
  def save(self):
   """Saves processed data."""


# 2.先加载小的数据集进行测试
# 3. 需要预先检测数据中是否存在None值
# 4. 查看数据处理（程序运行）进度
from tqdm import tqdm
import time


from fastprogress.fastprogress import master_bar,progress_bar
def check_process_1():
    mb=master_bar(range(10))
    for i in mb:
        for j in progress_bar(range(100),parent=mb):
            time.sleep(0.01)
            mb.child.comment=f"second bar start"
        mb.write(f"Finish Loop {i}")

# 5. pandas计算慢的问题，换成modin.pandas
import modin.pandas as pd

# 6. 使用装饰器记录运行时间
def timing(f):
    """Decorator for timing function"""
    # @wraps
    def wrapper(*args,**kwargs):
        start=time.time()
        result=f(*args,**kwargs)
        end=time.time()
        print("function %s took %2.2f sec"%(f.__name__,end-start))
        return result
    return wrapper

@timing
def check_process_0():
    text=""
    for char in tqdm([chr(ord('a')+i) for i in range(26)]):
        time.sleep(0.25)
        text=text+char

if __name__ == '__main__':
    # check_process_0()
    # check_process_1()
    timing(check_process_0())

