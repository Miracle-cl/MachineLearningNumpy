from __future__ import annotations

import pandas as pd
from collections import Counter
from dataclasses import dataclass, field
from typing import List
# from pprint import pprint
import math


# class TreeNode:
#     def __init__(self, feat_val, feature, label):
#         self.feat_val = feat_val # feat value of parent node
#         self.feature = feature # split feature
#         self.label = label # 1/0 if is_leaf else None
#         self.children = []
        
#     def __repr__(self):
#         return f'feat_val: {self.feat_val}, split_feature: {self.feature}, label: {self.label}'


@dataclass
class TreeNode:
    feat_val: str
    feature: str
    label: int
    children: List[TreeNode] = field(default_factory=list)


class PurningDT:
    # Reduced-Error Pruning
    def __init__(self, train_df, val_df, algo='ID3', epsilon=0.01):
        self.df = train_df
        self.val_df = val_df
        self.feat_val_mp = {col: set(self.df[col]) for col in self.df.columns}
        self.label = train_df.columns[-1]
        self.algo = algo
        self.epsilon = epsilon
        
    @staticmethod
    def info_entropy(df, label):
        # cal infomation entropy
        n = len(df)
        pks = [c / n for _, c in Counter(df[label]).items()]
        ent = - sum(pk * math.log2(pk) for pk in pks)
        return ent

    def gain(self, df, feat, label):
        # cal infomation gain
        d = len(df)
        ent = self.info_entropy(df, label) # infomation entropy
        condi_ent = 0 # conditional entropy
        V = set(df[feat])
        for val in V:
            _ent = self.info_entropy(df[df[feat] == val], label)
            dv = sum(df[feat] == val)
            # print(dv, d, _ent)
            condi_ent += _ent * dv / d 
        return ent - condi_ent
    
    @staticmethod
    def intrinsic_value(df, feat):
        # cal intrinsic value
        n = len(df)
        freqs = [c / n for c in df[feat].value_counts()]
        # if len(freqs) == 1:
        #     print(df)
        # print(feat, freqs, n)
        iv = - sum(pk * math.log2(pk) for pk in freqs)
        return iv

    def gain_ratio(self, df, feat, label):
        # cal infomation gain ratio
        info_gain = self.gain(df, feat, label)
        iv = self.intrinsic_value(df, feat)
        if iv == 0:
            return 0
        return info_gain / iv
    
    def feat_choose(self, df, feats, label):
        best_ = 0
        best_feat = None
        
        # feat_gain = [(feat, self.gain(df, feat, label)) for feat in feats]
        # feat_gain.sort(key=lambda x: x[1], reverse=True)
        # print(feat_gain)
        for feat in feats:
            if self.algo == 'ID3':
                info_gain = self.gain(df, feat, label)
            else: # elif self.algo == 'C4.5': simple
                info_gain = self.gain_ratio(df, feat, label)

            if info_gain > best_:
                best_ = info_gain
                best_feat = feat
        return best_, best_feat

    def feat_choose_c45(self, df, feats, label):
        # heuristic algorithm
        best_ = 0
        best_feat = None
        feat_gain = [(feat, self.gain(df, feat, label)) for feat in feats]
        feat_gain.sort(key=lambda x: x[1], reverse=True)
        for i in range(len(feat_gain) // 2):
            feat = feat_gain[i][0]
            info_gain_ratio = self.gain_ratio(df, feat, label)
            if info_gain_ratio > best_:
                best_ = info_gain_ratio
                best_feat = feat
        return best_, best_feat

    def train(self, df, val_df, feat_val, best_feat=None):
        # dataset is blank, then max freqency in all dataset
        if len(df) == 0:
            # print('00', best_feat, feat_val)
            max_freq_label = self.df[self.df[best_feat] == feat_val][self.label].value_counts().index[0]
            return TreeNode(feat_val, None, max_freq_label)
        
        feats, label = df.columns[:-1], df.columns[-1]
        y_train = df[label]

        # if all data belong one class then tree finished growing
        if len(y_train.value_counts()) == 1:
            # print('11', feat_val, y_train.iloc[0])
            return TreeNode(feat_val, None, y_train.iloc[0])

        # feats is blank, then choose max class as leaf label
        if len(feats) == 0:
            # print('22')
            return TreeNode(feat_val, None, y_train.value_counts(sort=True, ascending=False).index[0])

        if self.algo == 'C4.5':
            best_gain, best_feat = self.feat_choose_c45(df, feats, label)
        else:
            best_gain, best_feat = self.feat_choose(df, feats, label)

        if best_gain < self.epsilon:
            # print('33')
            return TreeNode(feat_val, None, y_train.value_counts(sort=True, ascending=False).index[0])
        
        # check if acc is increasing
        # choose max freq label as predict
        pred = y_train.value_counts(sort=True, ascending=False).index[0] # cal most freq label in df (train)
        corr = len(val_df[val_df[self.label] == pred]) # num of pred in val_df 
        tot = len(val_df)
        ori_acc = corr / tot

        new_acc = self.cal_corr(df, val_df, best_feat) / tot
        # print(best_feat, pred, ori_acc, new_acc)
        if new_acc <= ori_acc:
            # not split
            return TreeNode(feat_val, None, y_train.value_counts(sort=True, ascending=False).index[0])

        # need split
        root = TreeNode(feat_val, best_feat, None)

        for sub_feat_val in self.feat_val_mp[best_feat]:
            sub_df = df[df[best_feat] == sub_feat_val].drop(best_feat, axis=1)
            sub_val_df = val_df[val_df[best_feat] == sub_feat_val].drop(best_feat, axis=1)
            sub_tree = self.train(sub_df, sub_val_df, sub_feat_val, best_feat)
            root.children.append(sub_tree)
        return root
    
    def cal_corr(self, df, val_df, best_feat):
        corr= 0
        for f_val in self.feat_val_mp[best_feat]:
            feat_y = df[df[best_feat] == f_val][self.label]
            if len(feat_y) == 0:
                continue
            pred = feat_y.value_counts().index[0]
            # print(f_val, pred)
            corr += len(val_df[(val_df[best_feat] == f_val) & (val_df[self.label] == pred)])
        return corr

    def predict(self, root, val_df):
        def recurve(root, feat_vals):
            if not root.children:
                return root.label
            for child in root.children:
                if feat_vals[root.feature] == child.feat_val:
                    return recurve(child, feat_vals)
            return None

        corr = 0
        preds = []
        for _, item in val_df.iterrows():
            label = item[self.label]
            pred = recurve(root, item)
            corr += int(pred == label)
            # print(index, recurve(root, item), label)
            preds.append(pred)
        acc = corr / len(val_df)
        return preds, acc

    def dfs(self, root):
        # serialization of root
        if not root:
            return
        tree = {}
        tree['feat_val'] = root.feat_val
        tree['feature'] = root.feature
        tree['label'] = root.label
        tree['children'] = [self.dfs(child) for child in root.children]
        return tree


if __name__ == '__main__':
    dataset=[['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460,1],
            ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑',0.774,0.376,1],
            ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑',0.634,0.264,1],
            ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑',0.608,0.318,1],
            ['浅白','蜷缩','浊响','清晰','凹陷','硬滑',0.556,0.215,1],
            ['青绿','稍蜷','浊响','清晰','稍凹','软粘',0.403,0.237,1],
            ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘',0.481,0.149,1],
            ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑',0.437,0.211,1],

            ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑',0.666,0.091,0],
            ['青绿','硬挺','清脆','清晰','平坦','软粘',0.243,0.267,0],
            ['浅白','硬挺','清脆','模糊','平坦','硬滑',0.245,0.057,0],
            ['浅白','蜷缩','浊响','模糊','平坦','软粘',0.343,0.099,0],
            ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑',0.639,0.161,0],
            ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑',0.657,0.198,0],
            ['乌黑','稍蜷','浊响','清晰','稍凹','软粘',0.360,0.370,0],
            ['浅白','蜷缩','浊响','模糊','平坦','硬滑',0.593,0.042,0],
            ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑',0.719,0.103,0]]
    # cols = ['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率','好瓜']

    datasets = [row[:6] + row[-1:] for row in dataset]
    cols = ['色泽','根蒂','敲声','纹理','脐部','触感','好瓜']
    train_ids = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16, 4]
    val_ids = [3,4,7,8,10,11,12]

    df = pd.DataFrame.from_records(datasets, columns=cols)
    train_df = df.iloc[train_ids]
    val_df = df.iloc[val_ids]
    # print(train_df)

    dt = PurningDT(train_df, val_df, algo='C4.5')
    root = dt.train(train_df, val_df, None)
    print(dt.dfs(root))
    # root
    print('----------------------')
    acc, preds = dt.predict(root, val_df)
    print(acc, preds)