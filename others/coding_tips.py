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

# 7. 使用异常检测函数加测异常并做出相应处理
import os
def run_command(cmd):
    return os.system(cmd)

def shutdown(seconds=0,os_="linux"):
    if os_=="linux":
        run_command('sudo shutdown -h -t sec %s'%seconds)
    elif os_=="windows":
        run_command("shutdown -s -t %s"%seconds)

# 8. create and keep document,创建和保存报告
import json
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,fbeta_score)


def get_metrics(y,y_pred,beta=2,average_method="macro",y_encoder=None):
    if y_encoder:
        y=y_encoder.inverse_transfrom(y)
        y_pred=y_encoder.inverse_transfrom(y_pred)
        return {
            "accuracy":round(accuracy_score(y,y_pred),4),
            "f1_score_macro":round(f1_score(y,y_pred,average=average_method),4),
            "fbeta_score_macro":round(fbeta_score(y,y_pred,beta=beta,average=average_method),4),
            "report":classification_report(y,y_pred,output_dict=True),
            "report_csv":classification_report(y,y_pred,output_dict=False).replace('\n','\r\n')
        }
def save_metrics(metrics: dict, model_directory, file_name):
    path = os.path.join(model_directory, file_name + '_report.txt')
    classification_report_to_csv(metrics['report_csv'], path)
    metrics.pop('report_csv')
    path = os.path.join(model_directory, file_name + '_metrics.json')
    json.dump(metrics, open(path, 'w'), indent=4)

def classification_report_to_csv(metrics, path):
    pass

if __name__ == '__main__':
    # check_process_0()
    # check_process_1()
    timing(check_process_0())

