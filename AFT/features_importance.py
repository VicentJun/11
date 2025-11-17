import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import sklearn.inspection as sk_inspection
import shap
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    使用多种方法计算特征重要性并综合评估
    """
    
    def __init__(self, seed=42, n_splits=5, n_repeats=5):
        """
        Parameters:
        -----------
        seed : int
            随机种子
        n_splits : int
            交叉验证折数
        n_repeats : int
            Permutation Importance重复次数
        """
        self.SEED = seed
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.results = {}
        
    def _get_lightgbm_estimator(self):
        """LGBM和XGBOOST参数我随便设的，需要改"""
        return lgb.LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.1,
            n_estimators=100,
            subsample_for_bin=200000,
            objective='mae',
            class_weight=None,
            min_split_gain=0.0,
            min_child_weight=0.001,
            min_child_samples=20,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=self.SEED,
            n_jobs=-1,
            importance_type='gain',
            force_row_wise=True
        )
    
    def _get_xgboost_estimator(self):
        return xgb.XGBRegressor(
            objective='reg:absoluteerror',
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=self.SEED,
            n_jobs=-1
        )
    
    def calculate_lightgbm_importance(self, X_train, y_train):
        """
        计算LightGBM特征重要性
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练目标
            
        Returns:
        --------
        dict : 包含重要性结果和验证分数的字典
        """
        print("计算LightGBM特征重要性...")
        estimator = self._get_lightgbm_estimator()
        
        val_scores = []
        gain_importance_list = []
        split_importance_list = []
        
        # 使用时间序列交叉验证
        splitter = sk_model_selection.TimeSeriesSplit(n_splits=self.n_splits).split(X_train, y_train)
        
        for fold, (train_idx, val_idx) in enumerate(splitter):
            model = clone(estimator)
            
            # 划分训练集和验证集
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # 训练模型
            model.fit(X_train_fold, y_train_fold)
            
            # 获取两种重要性
            gain_importance_list.append(
                pd.Series(index=X_train.columns, data=model.feature_importances_)
            )
            
            # 计算split重要性
            model.set_params(importance_type='split')
            model.fit(X_train_fold, y_train_fold)
            split_importance_list.append(
                pd.Series(index=X_train.columns, data=model.feature_importances_)
            )
            
            # 验证分数
            val_score = sk_metrics.mean_absolute_error(
                model.predict(X_val_fold), y_val_fold
            )
            val_scores.append(val_score)
        
        # 计算平均重要性
        gain_importance = pd.concat(gain_importance_list, axis=1).mean(axis=1)
        split_importance = pd.concat(split_importance_list, axis=1).mean(axis=1)
        
        result = {
            'gain_importance': gain_importance,
            'split_importance': split_importance,
            'val_scores': val_scores,
            'mean_val_score': np.mean(val_scores)
        }
        
        print(f"LightGBM验证分数: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
        return result
    
    def calculate_xgboost_importance(self, X_train, y_train):
        """
        计算XGBoost特征重要性
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练目标
            
        Returns:
        --------
        dict : 包含重要性结果和验证分数的字典
        """
        print("计算XGBoost特征重要性...")
        estimator = self._get_xgboost_estimator()
        
        val_scores = []
        weight_importance_list = []
        gain_importance_list = []
        cover_importance_list = []
        
        splitter = sk_model_selection.TimeSeriesSplit(n_splits=self.n_splits).split(X_train, y_train)
        
        for fold, (train_idx, val_idx) in enumerate(splitter):
            model = clone(estimator)
            
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            
            # 获取不同类型的特征重要性
            importance_dict = model.get_booster().get_score(importance_type='weight')
            weight_importance_list.append(
                pd.Series(importance_dict).reindex(X_train.columns, fill_value=0)
            )
            
            importance_dict = model.get_booster().get_score(importance_type='gain')
            gain_importance_list.append(
                pd.Series(importance_dict).reindex(X_train.columns, fill_value=0)
            )
            
            importance_dict = model.get_booster().get_score(importance_type='cover')
            cover_importance_list.append(
                pd.Series(importance_dict).reindex(X_train.columns, fill_value=0)
            )
            
            val_score = sk_metrics.mean_absolute_error(
                model.predict(X_val_fold), y_val_fold
            )
            val_scores.append(val_score)
        
        # 计算平均重要性
        weight_importance = pd.concat(weight_importance_list, axis=1).mean(axis=1)
        gain_importance = pd.concat(gain_importance_list, axis=1).mean(axis=1)
        cover_importance = pd.concat(cover_importance_list, axis=1).mean(axis=1)
        
        result = {
            'weight_importance': weight_importance,
            'gain_importance': gain_importance,
            'cover_importance': cover_importance,
            'val_scores': val_scores,
            'mean_val_score': np.mean(val_scores)
        }
        
        print(f"XGBoost验证分数: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
        return result
    
    def calculate_permutation_importance(self, X_train, y_train, model_type='lightgbm'):
        """
        计算Permutation特征重要性
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练目标
        model_type : str
            模型类型 ('lightgbm' 或 'xgboost')，permutation重要性与model无关，选择其一即可
            
        Returns:
        --------
        pd.Series : Permutation重要性
        """
        print(f"计算{model_type}的Permutation重要性...")
        
        if model_type == 'lightgbm':
            estimator = self._get_lightgbm_estimator()
        else:
            estimator = self._get_xgboost_estimator()
        
        permutation_importance_list = []
        val_scores = []
        
        splitter = sk_model_selection.TimeSeriesSplit(n_splits=self.n_splits).split(X_train, y_train)
        
        for fold, (train_idx, val_idx) in enumerate(splitter):
            model = clone(estimator)
            
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            
            # 计算Permutation Importance
            result = sk_inspection.permutation_importance(
                model, X_val_fold, y_val_fold, 
                n_repeats=self.n_repeats, 
                random_state=self.SEED,
                scoring='neg_mean_absolute_error'
            )
            
            permutation_importance_list.append(
                pd.Series(index=X_train.columns, data=result.importances_mean)
            )
            
            val_score = sk_metrics.mean_absolute_error(
                model.predict(X_val_fold), y_val_fold
            )
            val_scores.append(val_score)
        
        # 计算平均Permutation重要性
        permutation_importance = pd.concat(permutation_importance_list, axis=1).mean(axis=1)
        
        print(f"{model_type} Permutation验证分数: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
        return permutation_importance
    
    def calculate_shap_importance(self, X_train, y_train, model_type='lightgbm', sample_size=1000):
        """
        计算SHAP特征重要性
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练目标
        model_type : str
            模型类型 ('lightgbm' 或 'xgboost')
        sample_size : int
            用于SHAP计算的样本数量
            
        Returns:
        --------
        pd.Series : SHAP重要性
        """
        print(f"计算{model_type}的SHAP重要性...")
        
        if model_type == 'lightgbm':
            estimator = self._get_lightgbm_estimator()
        else:
            estimator = self._get_xgboost_estimator()
        
      
        if len(X_train) > sample_size:
            # 对大数据集进行采样
            sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_idx]
            y_sample = y_train.iloc[sample_idx]
        else:
            X_sample = X_train
            y_sample = y_train
        
        model = clone(estimator)
        model.fit(X_sample, y_sample)
        
        # 计算SHAP值
        if model_type == 'lightgbm':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_sample)
        
        # 计算平均绝对SHAP值作为特征重要性
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_importance = pd.Series(
            index=X_train.columns,
            data=np.abs(shap_values).mean(axis=0)
        )
        
        return shap_importance
    
    def compute_all_importance(self, X_train, y_train):
        """
        计算所有特征重要性方法
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练目标
            
        Returns:
        --------
        dict : 包含所有重要性结果的字典
        """
        print("开始计算所有特征重要性方法...")
        
        # LightGBM重要性
        lgb_results = self.calculate_lightgbm_importance(X_train, y_train)
        self.results['lightgbm'] = lgb_results
        
        # XGBoost重要性
        xgb_results = self.calculate_xgboost_importance(X_train, y_train)
        self.results['xgboost'] = xgb_results
        
        # Permutation重要性
        self.results['permutation_lgb'] = self.calculate_permutation_importance(
            X_train, y_train, 'lightgbm'
        )
        self.results['permutation_xgb'] = self.calculate_permutation_importance(
            X_train, y_train, 'xgboost'
        )
        
        # SHAP重要性
        self.results['shap_lgb'] = self.calculate_shap_importance(
            X_train, y_train, 'lightgbm'
        )
        self.results['shap_xgb'] = self.calculate_shap_importance(
            X_train, y_train, 'xgboost'
        )
        
        print("所有特征重要性计算完成!")
        return self.results
    
    def get_combined_importance(self, top_k=300):
        """
        综合所有方法得到最终特征重要性排名
        
        Parameters:
        -----------
        top_k : int
            要选择的特征数量
            
        Returns:
        --------
        pd.DataFrame : 包含综合排名和所有方法得分的DataFrame
        """
        if not self.results:
            raise ValueError("请先运行 compute_all_importance 方法")
        
        # 收集所有重要性分数
        importance_data = {}
        
        # LightGBM
        importance_data['lgb_gain'] = self.results['lightgbm']['gain_importance']
        importance_data['lgb_split'] = self.results['lightgbm']['split_importance']
        
        # XGBoost
        importance_data['xgb_weight'] = self.results['xgboost']['weight_importance']
        importance_data['xgb_gain'] = self.results['xgboost']['gain_importance']
        importance_data['xgb_cover'] = self.results['xgboost']['cover_importance']
        
        # Permutation
        importance_data['perm_lgb'] = self.results['permutation_lgb']
        importance_data['perm_xgb'] = self.results['permutation_xgb']
        
        # SHAP
        importance_data['shap_lgb'] = self.results['shap_lgb']
        importance_data['shap_xgb'] = self.results['shap_xgb']
        
        # 创建综合DataFrame
        combined_df = pd.DataFrame(importance_data)
        
        # 归一化所有重要性分数
        normalized_df = combined_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        # 计算综合得分，使用加权平均的，权重设置需要额外考虑，这里是随便设置的
        weights = {
            'perm_lgb': 1.5, 'perm_xgb': 1.5,
            'shap_lgb': 1.3, 'shap_xgb': 1.3,
            'lgb_gain': 1.0, 'xgb_gain': 1.0,
            'lgb_split': 0.8, 'xgb_weight': 0.8, 'xgb_cover': 0.8
        }
        
        for col in normalized_df.columns:
            if col in weights:
                normalized_df[col] = normalized_df[col] * weights[col]
        
        # 计算平均综合得分
        combined_df['composite_score'] = normalized_df.mean(axis=1)
        combined_df['final_rank'] = combined_df['composite_score'].rank(ascending=False)
        
        # 排序并返回top_k
        final_ranking = combined_df.sort_values('composite_score', ascending=False)
        
        print(f"综合特征选择完成，前{top_k}个特征已确定")
        return final_ranking.head(top_k)


if __name__ == "__main__":
    cross_section_features = pd.read_parquet(
    "cross_section_features.parquet",
    engine="pyarrow",
    use_nullable_dtypes=False
    ).iloc[:100000]

    X_train = cross_section_features.query("target.notna()").drop(['row_id'], axis=1)
    y_train = X_train.pop("target")

    cols_contain_inf = (
    X_train
    .select_dtypes("float")
    .columns
    [
        np.isinf(
            X_train
            .select_dtypes("float")
        ).sum()!=0
    ]
    )

    X_train[cols_contain_inf] = (
        X_train
        [cols_contain_inf]
        .mask(
            np.isinf(X_train[cols_contain_inf])
        )
    )

    
    
    analyzer = FeatureImportanceAnalyzer(seed=42, n_splits=5, n_repeats=5)
    results = analyzer.compute_all_importance(X_train, y_train)
    top_features = analyzer.get_combined_importance(top_k=300)
    
    print("前10个最重要的特征:")
    print(top_features.head(10))
    
    # 保存结果
    top_features.to_csv('feature_importance_results.csv')
    
    # 获取最终选择的特征名
    selected_features = top_features.index.tolist()
    print(f"\n已选择 {len(selected_features)} 个特征")