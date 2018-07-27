# ali_nlp_competition
B榜，第54名，总共600支队伍，进入复赛
主函数：main.py
jieba词：jieba
停止词：chinese_stopwords
相似词：similar_words
程序运行说明：
线下测试：
  将所有的SUBMIT改为False;
  将最后预测部分注释，将深度学习训练部分注释去掉;
  分为10折训练，每次都会有一个.h5的训练模型保存下来;
  预测模型中：将每个模型的预测结果进行加权平均。
线上提交：
  直接提交run.bash即可。
