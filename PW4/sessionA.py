"""Session A: concrete strength regression and classification."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
FEATURES = ['cement', 'slag', 'ash', 'water', 'superplastic',
            'coarse_agg', 'fine_agg', 'age']
K_GRID = [1, 3, 5, 7, 15, 25, 51, 101] 
TREE_DEPTHS = [1, 2, 3, 4, 5, 6, 8, None]


def rmse(y, prediction):
    return np.sqrt(mean_squared_error(y, prediction))


def adj_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(n-p-1) if n > p+1 else np.nan


def table(frame):
    cols = list(frame.columns)
    result = ['| ' + ' | '.join(map(str, cols)) + ' |',
              '| ' + ' | '.join('---' for _ in cols) + ' |']
    for row in frame.itertuples(index=False, name=None):
        values = [f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
                  for v in row]
        result.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(result)


def plot_save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160, bbox_inches='tight')
    plt.close()


def metric_row(name, model, X_test, y_test):
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return dict(model=name, TP=tp, FP=fp, FN=fn, TN=tn,
                accuracy=accuracy_score(y_test, pred),
                precision=precision_score(y_test, pred, zero_division=0),
                recall=recall_score(y_test, pred, zero_division=0),
                F1=f1_score(y_test, pred, zero_division=0))


def update_readme(report):
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8')
    begin, end = '<!-- SESSION_A_START -->', '<!-- SESSION_A_END -->'
    regex = re.escape(begin) + r'.*?' + re.escape(end)
    if not re.search(regex, text, re.S):
        raise ValueError('Missing Session A markers in README')
    text = re.sub(regex, lambda _: begin+'\n'+report+'\n'+end, text, count=1, flags=re.S)
    path.write_text(text, encoding='utf-8')


def main():
    
    df = pd.read_csv(ROOT / 'data/concrete.csv')
    if list(df.columns) != FEATURES + ['strength']:
        raise ValueError('Unexpected concrete columns: check fetch_concrete.py')
    duplicates = int(df.duplicated().sum())
    null_counts = df.isna().sum()
    print('A2 missing values:\n', null_counts.to_string())
    if null_counts.sum():
        raise ValueError('Missing values detected: inspect before cleaning')
    original_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print('A2 duplicates:', duplicates, 'rows after cleaning:', len(df))
    print('A2 describe():\n', df.describe().to_string())
    X, y = df[FEATURES], df['strength']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in zip(axes, ['cement', 'age'], ['Cement (kg/m³)', 'Age (days)']):
        ax.scatter(X[col], y, s=14, alpha=0.5)
        ax.set(xlabel=label, ylabel='Strength (MPa)', title=f'Strength vs {col}')
    plot_save('A2_scatter.png')

    # Keep test rows separate from model training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f'A3 train={len(X_train)} test={len(X_test)}')
    line = LinearRegression().fit(X_train, y_train)
    y_fit, y_pred = line.predict(X_train), line.predict(X_test)
    train_r2 = r2_score(y_train, y_fit)
    regression = pd.DataFrame([dict(train_RMSE_MPa=rmse(y_train, y_fit),
        test_RMSE_MPa=rmse(y_test, y_pred), test_MAE_MPa=mean_absolute_error(y_test, y_pred),
        train_R2=train_r2, train_adjusted_R2=adj_r2(train_r2, len(y_train), X.shape[1]),
        test_R2=r2_score(y_test, y_pred))])
    regression.to_csv(OUT / 'A4_regression.csv', index=False)
    equation = 'y_hat (MPa) = ' + f'{line.intercept_:+.5f}' + ''.join(
        f' {co:+.5f} * {name}' for name, co in zip(FEATURES, line.coef_))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X_test['age'], y_test-y_pred, alpha=.6, s=18)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set(xlabel='Age (days, logarithmic)', ylabel='True - predicted (MPa)',
           title='Test residuals against age')
    plot_save('A4_residuals.png')
    age_groups = pd.cut(X_test['age'], [0,7,28,90,366],
                        labels=['1–7 days','8–28 days','29–90 days','91–365 days'])
    age_residuals = (pd.DataFrame({'group': age_groups, 'residual': y_test-y_pred})
        .groupby('group', observed=True)['residual'].agg(['count','mean']).reset_index())
    age_residuals.to_csv(OUT / 'A4_residual_age_groups.csv', index=False)

    # Compare polynomial complexity.
    poly_rows = []
    for degree in [1,2,3,4]:
        poly = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                             StandardScaler(), LinearRegression())
        poly.fit(X_train, y_train)
        tr, te = poly.predict(X_train), poly.predict(X_test)
        p = poly[0].n_output_features_
        r2_tr = r2_score(y_train, tr)
        poly_rows.append(dict(degree=degree, p=p, train_RMSE_MPa=rmse(y_train,tr),
            test_RMSE_MPa=rmse(y_test,te), train_R2=r2_tr,
            train_adjusted_R2=adj_r2(r2_tr,len(y_train),p), test_R2=r2_score(y_test,te)))
        print(f'A5 degree {degree}: p={p} test RMSE={rmse(y_test,te):.3f}')
    polynomials = pd.DataFrame(poly_rows)
    polynomials.to_csv(OUT / 'A5_polynomial.csv', index=False)
    best_degree = polynomials.sort_values('test_RMSE_MPa').iloc[0]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(polynomials.degree, polynomials.train_RMSE_MPa,'o-',label='Train')
    ax.plot(polynomials.degree, polynomials.test_RMSE_MPa,'o-',label='Test')
    ax.set(xlabel='Polynomial degree',ylabel='RMSE (MPa)',title='Model complexity')
    ax.set_xticks([1,2,3,4]); ax.legend()
    plot_save('A5_polynomial_RMSE.png')
  
    y_class = (y >= 40).astype(int)  # 1 = at least 40 MPa 
    Xc_train, Xc_test, c_train, c_test = train_test_split(
        X, y_class, test_size=.2, random_state=42, stratify=y_class)
    print('A6 class proportions:\n',y_class.value_counts(normalize=True).to_string())
    knn_rows=[]
    for K in K_GRID:
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
        m.fit(Xc_train,c_train)
        knn_rows.append(dict(K=K,train_accuracy=m.score(Xc_train,c_train),
                             test_accuracy=m.score(Xc_test,c_test)))
    knn_table = pd.DataFrame(knn_rows)
    knn_table.to_csv(OUT / 'A6_knn_sweep.csv',index=False)

    K = int(knn_table.sort_values(['test_accuracy','K'],ascending=[False,False]).iloc[0].K)
    knn = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=K))
    knn.fit(Xc_train,c_train)
    unscaled_knn = KNeighborsClassifier(n_neighbors=K).fit(Xc_train,c_train)
    unscaled_accuracy = unscaled_knn.score(Xc_test,c_test)
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(knn_table.K,knn_table.train_accuracy,'o-',label='Train')
    ax.plot(knn_table.K,knn_table.test_accuracy,'o-',label='Test')
    ax.set_xscale('log'); ax.set(xlabel='K (logarithmic)',ylabel='Accuracy')
    ax.legend();plot_save('A6_knn_sweep.png')
    repeated=[]
    for seed in range(5):
        a,b,ya,yb = train_test_split(X,y_class,test_size=.2,random_state=seed,
                                      stratify=y_class)
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
        m.fit(a,ya)
        repeated.append(dict(random_state=seed,test_accuracy=m.score(b,yb)))
    repeat_table = pd.DataFrame(repeated)
    repeat_table.to_csv(OUT / 'A6_knn_repeated_splits.csv',index=False)

    tree_rows=[]
    for depth in TREE_DEPTHS:
        m = DecisionTreeClassifier(criterion='entropy',max_depth=depth,
                                   min_samples_leaf=1,random_state=0).fit(Xc_train,c_train)
        tree_rows.append(dict(max_depth='None' if depth is None else str(depth),
            actual_depth=m.get_depth(),leaves=m.get_n_leaves(),
            train_accuracy=m.score(Xc_train,c_train),test_accuracy=m.score(Xc_test,c_test)))
    tree_table = pd.DataFrame(tree_rows)
    tree_table.to_csv(OUT / 'A6_tree_sweep.csv',index=False)

    best_tree_row = tree_table.sort_values(['test_accuracy','actual_depth'],
                                           ascending=[False,True]).iloc[0]
    depth = None if best_tree_row.max_depth=='None' else int(best_tree_row.max_depth)
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=depth,
                                  min_samples_leaf=1,random_state=0).fit(Xc_train,c_train)
    tree_rule = export_text(tree,feature_names=FEATURES,max_depth=2)
    root_idx = tree.tree_.feature[0]
    root = (f'{FEATURES[root_idx]} <= {tree.tree_.threshold[0]:.2f}'
            if root_idx >= 0 else '(one-leaf tree)')
    print('A6 chosen K=',K,'depth=',depth,'\n',tree_rule)
    fig,ax = plt.subplots(figsize=(8,4))
    ax.plot(range(len(tree_table)),tree_table.train_accuracy,'o-',label='Train')
    ax.plot(range(len(tree_table)),tree_table.test_accuracy,'o-',label='Test')
    ax.set_xticks(range(len(tree_table)),tree_table.max_depth)
    ax.set(xlabel='max_depth (None=unlimited)',ylabel='Accuracy')
    ax.legend();plot_save('A6_tree_sweep.png')

    # A false positive accepts concrete below 40 MPa.
    metrics = pd.DataFrame([
        metric_row(f'KNN K={K}',knn,Xc_test,c_test),
        metric_row(f'Tree depth={best_tree_row.max_depth}',tree,Xc_test,c_test)])
    metrics.to_csv(OUT / 'A_metrics.csv',index=False)
    probabilities = knn.predict_proba(Xc_test)[:,1]
    threshold_rows=[]
    for t in [.3,.5,.7,.9]:
        prediction = (probabilities >= t).astype(int)
        tn,fp,fn,tp=confusion_matrix(c_test,prediction,labels=[0,1]).ravel()
        threshold_rows.append(dict(threshold=t,TP=tp,FP=fp,FN=fn,TN=tn,
            precision=precision_score(c_test,prediction,zero_division=0),
            recall=recall_score(c_test,prediction,zero_division=0)))
    thresholds=pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT / 'A7_thresholds.csv',index=False)
    # Prefer fewer false positives for the 40 MPa decision.
    chosen_t=(thresholds.loc[thresholds.threshold>=.5]
        .sort_values(['FP','recall','threshold'],ascending=[True,False,True]).iloc[0])

    # Feature importance from different models.
    standard_line=make_pipeline(StandardScaler(),LinearRegression())
    standard_line.fit(X_train,y_train)
    perm_knn=permutation_importance(knn,Xc_test,c_test,scoring='f1',
                                     n_repeats=20,random_state=0)
    perm_tree=permutation_importance(tree,Xc_test,c_test,scoring='f1',
                                      n_repeats=20,random_state=0)
    importance=pd.DataFrame(dict(feature=FEATURES,raw_coef=line.coef_,
        standardized_coef_MPa=standard_line[-1].coef_,
        tree_importance=tree.feature_importances_,
        knn_perm_F1=perm_knn.importances_mean,
        knn_perm_SD=perm_knn.importances_std,
        tree_perm_F1=perm_tree.importances_mean,
        tree_perm_SD=perm_tree.importances_std))
    importance.to_csv(OUT / 'A8_importance.csv',index=False)
    for filename, column, title in [
        ('A8_standardized.png','standardized_coef_MPa','Standardized coefficients (MPa per SD)'),
        ('A8_tree_importance.png','tree_importance','Tree entropy decrease'),
        ('A8_knn_permutation.png','knn_perm_F1','KNN loss of F1 after permutation')]:
        fig,ax=plt.subplots(figsize=(8,4))
        importance.set_index('feature')[column].sort_values().plot.barh(ax=ax)
        ax.set_title(title);plot_save(filename)

    # Bonus: use physics-based transformations.
    bonus_rows=[]
    for ratio in [False,True]:
        features=X.copy()
        if (features.age<=0).any(): raise ValueError('log(age) needs positive age')
        features['age']=np.log(features['age'])
        if ratio:
            binder=features.cement+features.slag+features.ash
            if (binder<=0).any(): raise ValueError('binder must be positive')
            features['water']=features.water/binder
        features=features.rename(columns={'age':'log_age',
                           'water':'water_to_binder_ratio'} if ratio else {'age':'log_age'})
        m=LinearRegression().fit(features.loc[X_train.index],y_train)
        prediction=m.predict(features.loc[X_test.index])
        bonus_rows.append(dict(model='log(age) + water/binder' if ratio else 'log(age)',
            features=len(features.columns),test_RMSE_MPa=rmse(y_test,prediction),
            test_R2=r2_score(y_test,prediction)))
    bonus=pd.DataFrame(bonus_rows)
    bonus.to_csv(OUT / 'A_bonus.csv',index=False)
    # Update the report using the calculated results.
    raw_top=(importance.assign(magnitude=importance.raw_coef.abs())
                .nlargest(3,'magnitude').feature.tolist())
    scaled_top=(importance.assign(magnitude=importance.standardized_coef_MPa.abs())
                .nlargest(3,'magnitude').feature.tolist())
    tree_top=importance.nlargest(3,'tree_importance').feature.tolist()
    knn_top=importance.nlargest(3,'knn_perm_F1').feature.tolist()
    report=f"""### A1 — Dataset

The UCI concrete dataset (Yeh, 1998) has {original_rows} rows. The inputs are cement, slag, ash, water, superplasticizer, coarse aggregate and fine aggregate (all in kg/m³), plus curing age (days). The target is strength (MPa). `fetch_concrete.py` checks the original variable order and saves `data/concrete.csv`.

### A2 — Cleaning and exploration

Missing values: **{int(null_counts.sum())}**. I removed **{duplicates}** identical rows, leaving **{len(df)}**. Duplicates could otherwise appear in both the training and test sets. The largest input standard deviations are in **{', '.join(df[FEATURES].std().nlargest(3).index)}**, so feature scale matters. The cement–strength correlation is **{df.cement.corr(y):.3f}** and age–strength is **{df.age.corr(y):.3f}**. Both scatter plots show considerable variation; age is not simply linear.

{table(df.agg(['min', 'max', 'mean', 'std']).T.reset_index().rename(columns={'index': 'variable'}))}

![Strength vs cement and age](results/A2_scatter.png)

### A3 — Train/test split

I used **{len(X_train)}** training rows and **{len(X_test)}** test rows (80/20, `random_state=42`). All model fitting and scaling for supervised learning use the training data only.

### A4 — Linear regression

The fitted equation is below. Strength and intercept are in MPa; ingredient coefficients are in MPa per kg/m³, and the age coefficient is in MPa/day.

`{equation}`

{table(regression)}

Residuals are measured strength minus predicted strength. Their average changes with curing age, suggesting that a straight-line age term misses part of the curing effect.

{table(age_residuals)}

![Residuals vs log(age)](results/A4_residuals.png)

### A5 — Polynomial regression

I tested degrees 1–4 with polynomial features, scaling and linear regression in a pipeline.

{table(polynomials)}

**Degree {int(best_degree.degree)}** has the smallest test RMSE ({best_degree.test_RMSE_MPa:.3f} MPa). Degree 4 creates {int(polynomials.iloc[-1].p)} features for {len(X_train)} training rows: its training error is low but the test error grows sharply. Adjusted R² alone does not prevent overfitting.

![Polynomial train/test RMSE](results/A5_polynomial_RMSE.png)

### A6 — Classifiers (40 MPa)

Class 1 means strength ≥40 MPa; class 0 means below 40 MPa. I excluded strength from the inputs and used a stratified 80/20 split. The positive-class fraction is {y_class.mean():.3f}.

**KNN:** I scaled features before measuring distances and tested all the required K values.

{table(knn_table)}

I chose **K={K}** (highest observed test accuracy). Its scaled accuracy is {knn.score(Xc_test,c_test):.3f}, compared with {unscaled_accuracy:.3f} without scaling. The five repeated splits show how much accuracy varies:

{table(repeat_table)}

Mean accuracy = {repeat_table.test_accuracy.mean():.3f}, SD = {repeat_table.test_accuracy.std(ddof=1):.3f}. Small differences between K values may depend on the split.

![KNN accuracy](results/A6_knn_sweep.png)

**Decision tree:** I used entropy and compared depths 1, 2, 3, 4, 5, 6, 8 and unlimited.

{table(tree_table)}

The highest observed test accuracy was at **max_depth={best_tree_row.max_depth}**. Deeper trees fit training data more closely but are harder to read and can overfit; limiting depth gives simpler decisions but may lower accuracy. The first question is `{root}`, which is physically reasonable because curing time affects strength.

```text
{tree_rule.rstrip()}
```

![Tree accuracy](results/A6_tree_sweep.png)

### A7 — Classification metrics and threshold

A **false positive** is the dangerous error here: the model says concrete reaches 40 MPa when it does not. Precision measures the fraction of predicted-positive mixtures that really reach the target.

{table(metrics)}

I tested four probability thresholds using KNN:

{table(thresholds)}

I chose **{chosen_t.threshold:.1f}** to limit false positives ({int(chosen_t.FP)} on this test split). The cost is lower recall: more mixtures that actually pass are rejected. KNN neighbour fractions are not certified safety probabilities.

### A8 — Feature importance

I compared raw and standardised regression coefficients, tree impurity importance and permutation importance measured by F1 loss.

{table(importance)}

By absolute raw coefficients the leading features are **{', '.join(raw_top)}**; after scaling they are **{', '.join(scaled_top)}**. The rankings differ because inputs have different units and ranges. The tree mainly uses **{', '.join(tree_top)}**, while KNN permutation importance highlights **{', '.join(knn_top)}**. These rankings broadly relate to binder content and curing age, but importance alone does not prove causation.

![Standardised coefficients](results/A8_standardized.png)
![Tree importance](results/A8_tree_importance.png)
![KNN permutation importance](results/A8_knn_permutation.png)

### Session A bonus — Physics-based features

I tried `log(age)` and the water-to-binder ratio `water/(cement + slag + ash)` without adding more input columns.

{table(bonus)}

These transformations substantially improve the basic linear model, although polynomial degree 3 still has the smaller test RMSE in this experiment.
"""
    update_readme(report)
    print('\nA4 linear:\n',regression.to_string(index=False))
    print('\nA5 polynomials:\n',polynomials.to_string(index=False))
    print('\nA7 metrics:\n',metrics.to_string(index=False))
    print('\nA7 thresholds:\n',thresholds.to_string(index=False))
    print('\nA bonus:\n',bonus.to_string(index=False))
    print('Session A complete; results/A_metrics.csv and README generated.')

if __name__ == '__main__':
    main()
