import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import orangecontrib.associate.fpgrowth as oaf
import functools

'''
对数据集进行处理，转换成适合进行关联规则挖掘的形式；
找出频繁模式；
导出关联规则，计算其支持度和置信度;
对规则进行评价，可使用Lift、卡方和其它教材中提及的指标, 至少2种；
对挖掘结果进行分析；
可视化展示。
data1:
country  designation  points  price  province  region_1  region_2  variety  winery
description  
data2: 
country  designation  points  price  province  region_1  region_2  variety  winery
description  taster_name  taster_twitter_handle  title
"country":葡萄酒所属国家
"description":关于葡萄酒的说明
"designation":葡萄的产地
"points":葡萄酒的评分
"price":一瓶葡萄酒的价格
"province":生产葡萄酒的省份
"region_1":葡萄种植地
"region_2":葡萄特定种植地
"taster_name":品酒师
"taster_twitter_handle":品酒师的推特
"title":葡萄酒评论的标题
"variety":用于制酒的葡萄的品种
"winery":酿酒厂

'''


if __name__ == "__main__":
    path1 = './Wine Reviews/winemag-data_first150k.csv'
    path2 = './Wine Reviews/winemag-data-130k-v2.csv'
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    # print(data1.isnull().sum())
    # print(data2.isnull().sum())

    data1.drop(['country', 'description', 'designation'], axis=1, inplace=True)
    data1.drop(data1.columns[0], axis=1, inplace=True)
    data1.info()
    data2.drop(['country', 'description', 'designation', 'taster_name', 'taster_twitter_handle', 'title'], axis=1,
               inplace=True)
    data2.drop(data2.columns[0], axis=1, inplace=True)
    data2.info()

    # 数据集合并，删除缺失值
    data = pd.concat([data1, data2], ignore_index=True)
    data.info()
    print(data.isnull().sum())
    # new_data = data.dropna(axis=0, subset=['region_2', 'country'])
    new_data = data.dropna(axis=0)
    # 在预处理数据后得到的data中，由于country属性中均为'US'，故之后又把country属性删去
    # print(new_data['country'].value_counts())
    # new_data.drop(['country'], axis=1, inplace=True)
    # 重置索引
    N = len(new_data)
    new_data.index = range(N)
    new_data.info()

    # country_count = new_data['country'].value_counts()
    # print("country:\n", country_count.index)
    # print(len(np.array(country_count.index)))

    listToAnalysis = []
    listToStore = []
    # country  designation  points  price  province  region_1  region_2  variety  winery
    for i in range(new_data.iloc[:, 0].size):
        temp = new_data.iloc[i]['points']
        listToStore.append(temp)
        temp = new_data.iloc[i]['price']
        if temp <= 50:
            temp = 'price_0_50'
        elif 50 < temp <= 100:
            temp = 'price_50_100'
        elif 100 < temp <= 200:
            temp = 'price_100_200'
        elif temp > 200:
            temp = 'price_200'
        listToStore.append(temp)
        temp = new_data.iloc[i]['province']
        listToStore.append(temp)
        temp = new_data.iloc[i]['region_2']
        listToStore.append(temp)
        temp = new_data.iloc[i]['variety']
        listToStore.append(temp)
        temp = new_data.iloc[i]['winery']
        listToStore.append(temp)
        listToAnalysis.append(listToStore.copy())
        listToStore.clear()

    strSet = set(functools.reduce(lambda a, b: a + b, listToAnalysis))
    strEncode = dict(zip(strSet, range(len(strSet))))
    strDecode = dict(zip(strEncode.values(), strEncode.keys()))
    listToAnalysis_int = [list(map(lambda item: strEncode[item], row)) for row in listToAnalysis]

    result = oaf.frequent_itemsets(listToAnalysis_int, .02)  # 支持度
    itemsets = dict(result)

    # 输出结果：[频繁项，支持度]
    items = []
    for i in itemsets:
        temStr = ''
        for j in i:
            temStr = temStr + str(strDecode[j]) + ' & '
        temStr = temStr[:-3]
        items.append([temStr, round(itemsets[i] / N, 4)])
        temStr = temStr + ': ' + str(round(itemsets[i] / N, 4))
        # print(temStr)
    # print(items)
    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame(items, columns=['频繁项集', '支持度'])
    df = df.sort_values('支持度', ascending=False)
    df.index = range(len(df))

    rules = oaf.association_rules(itemsets, .5)  # 置信度
    rules = list(rules)

    # Rules(规则前项，规则后项，支持度，置信度)
    returnRules = []
    for i in rules:
        temStr = ''
        for j in i[0]:  # 处理第一个frozenset
            temStr = temStr + str(strDecode[j]) + ' & '
        temStr = temStr[:-3]
        temStr = temStr + ' ==> '
        for j in i[1]:
            temStr = temStr + strDecode[j] + ' & '
        temStr = temStr[:-3]
        returnRules.append([temStr, round(i[2] / N, 4), round(i[3], 4)])
        temStr = temStr + ';' + '\t' + str(i[2]) + ';' + '\t' + str(i[3])
    df = pd.DataFrame(returnRules, columns=['关联规则', '支持度', '置信度'])
    df = df.sort_values('置信度', ascending=False)
    df.index = range(len(df))

    results = list(oaf.rules_stats(rules, itemsets, len(listToAnalysis)))
    resultsRules = []
    for i in results:
        temStr = ''
        for j in i[0]:  # 处理第一个frozenset
            temStr = temStr + str(strDecode[j]) + ' & '
        temStr = temStr[:-3]
        temStr = temStr + ' ==> '
        for j in i[1]:
            temStr = temStr + strDecode[j] + ' & '
        temStr = temStr[:-3]
        resultsRules.append([temStr, round(i[2] / N, 4), round(i[3], 4), round(i[6], 4), round(i[7], 4)])
        temStr = temStr + ';' + '\t' + str(i[2]) + ';' + '\t' + str(i[3]) + ';' + '\t' + str(i[6]) + ';' + '\t' + str(
            i[7])
    df = pd.DataFrame(resultsRules, columns=['关联规则', '支持度', '置信度', 'lift', 'leverage'])
    df = df.sort_values('lift', ascending=False)
    df.index = range(len(df))




