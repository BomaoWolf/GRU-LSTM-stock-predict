欢迎来到市场预测Api！
本接口会使用已训练好的模型来根据您的需求预测未来的市场数据
目前已有功能接口：
    /hello    打个招呼吧！ 无参数
    /predict    对现有的数据进行预测 无参数
    /trainAndPredict    训练一个专属于你的模型，RequestBody:code(股票代码,必须)、echos(训练字数)、butch_size(批大小) 例如:{"code":"601998.SH","echos":"100"} 除了会返回预测结果外，还会返回模型保存的路径
    /predictWithPath    使用已知的模型路径对实时的数据进行预测,RequestBody:path(模型路径,必须) 例如：{"path": "D:/model/api_saved/601998SH_100_64_20210512025931"}