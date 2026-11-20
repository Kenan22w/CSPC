"""Session B: PCA and clustering of wine chemistry data."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def table(df):
    cols=list(df.columns)
    lines=['| '+' | '.join(map(str,cols))+' |','| '+' | '.join('---' for _ in cols)+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating))
                else str(v) for v in row)+' |')
    return '\n'.join(lines)


def save(name):
    plt.tight_layout()
    plt.savefig(OUT/name,dpi=160,bbox_inches='tight')
    plt.close()


def update_readme(report):
    path=ROOT/'README.md'
    contents=path.read_text(encoding='utf-8')
    start,end='<!-- SESSION_B_START -->','<!-- SESSION_B_END -->'
    pattern=re.escape(start)+r'.*?'+re.escape(end)
    if not re.search(pattern,contents,flags=re.S):
        raise ValueError('Missing Session B markers in README')
    new=re.sub(pattern,lambda _:start+'\n'+report+'\n'+end,
               contents,count=1,flags=re.S)
    path.write_text(new,encoding='utf-8')


def cluster_agreement(true_labels, groups):
    """Agreement after matching each cluster to its most common cultivar."""
    cross=pd.crosstab(pd.Series(np.asarray(true_labels),name='cultivar'),
                      pd.Series(np.asarray(groups),name='cluster'))
    agree=float(cross.max(axis=0).sum()/len(true_labels))
    return agree,cross


def describe_cluster(scaled_center, columns):

    order=np.argsort(np.abs(scaled_center))[::-1][:3]
    pieces=[]
    for i in order:
        direction='higher' if scaled_center[i]>0 else 'lower'
        pieces.append(f'{direction} {columns[i]} ({scaled_center[i]:+.2f} SD)')
    return '; '.join(pieces)


def main():
    # Keep cultivar labels hidden during unsupervised learning.
    wine=load_wine(as_frame=True)
    X=wine.data
    cultivar=wine.target.copy()
    print('B1 wine.DESCR:\n',wine.DESCR)
    print('B1 describe():\n',X.describe().to_string())
    ranges=X.max()-X.min()
    deviations=X.std()

    # Compare PCA before and after scaling.
    pca_raw=PCA().fit(X)
    raw_loadings=pd.DataFrame({'feature':X.columns,
                  'PC1_raw_loading':pca_raw.components_[0]})
    raw_loadings=raw_loadings.sort_values('PC1_raw_loading',ascending=False)
    raw_loadings.to_csv(OUT/'B2_raw_PC1_loadings.csv',index=False)
    raw_dominant=raw_loadings.iloc[raw_loadings.PC1_raw_loading.abs().argmax()].feature


    scaler=StandardScaler()
    Z=scaler.fit_transform(X)
    print('B3 scaled means:\n',Z.mean(axis=0).round(5))
    print('B3 scaled SDs:\n',Z.std(axis=0).round(5))
    pca=PCA().fit(Z)
    variance=pca.explained_variance_ratio_
    cumulative=np.cumsum(variance)
    variance_table=pd.DataFrame({'PC':np.arange(1,len(variance)+1),
                                 'variance_ratio':variance,
                                 'cumulative_ratio':cumulative})
    variance_table.to_csv(OUT/'B3_variance.csv',index=False)
    components_90=int(np.searchsorted(cumulative,.9)+1)
    pc2_var=float(cumulative[1])
    loads=pd.DataFrame({'feature':X.columns,'PC1':pca.components_[0],
                        'PC2':pca.components_[1]})
    loads.to_csv(OUT/'B3_loadings.csv',index=False)
    pc1_sorted=loads.sort_values('PC1')
    pc2_sorted=loads.sort_values('PC2')

    fig,ax=plt.subplots(figsize=(9,4))
    x=np.arange(1,len(variance)+1)
    ax.bar(x,variance,label='Variance per component')
    ax.plot(x,cumulative,'o-',label='Cumulative variance')
    ax.axhline(.9,color='black',ls='--',lw=1,label='90% target')
    ax.set(xlabel='Principal component',ylabel='Share of explained variance',
           title='B3: Scree plot (scaled wine chemistry)')
    ax.set_xticks(x);ax.legend()
    save('B3_scree.png')
    Z_2d=pca.transform(Z)[:,:2]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(Z_2d[:,0],Z_2d[:,1],s=24,alpha=.65)
    ax.set(xlabel='PC1',ylabel='PC2',title='B3: All wines (labels hidden)')
    save('B3_unlabelled_PCA.png')

    # Compare random starting points and different cluster counts.
    seed_rows=[]
    for seed in range(8):
        km=KMeans(n_clusters=3,init='k-means++',n_init=1,
                  max_iter=300,random_state=seed).fit(Z)
        seed_rows.append({'seed':seed,'inertia':km.inertia_})
    seed_table=pd.DataFrame(seed_rows)
    seed_table.to_csv(OUT/'B4_initializations.csv',index=False)
    elbow_rows=[]
    for k in range(1,11):
        model=KMeans(n_clusters=k,init='k-means++',n_init=10,
                     max_iter=300,random_state=0).fit(Z)
        elbow_rows.append({'k':k,'inertia':model.inertia_})
    elbow=pd.DataFrame(elbow_rows)
    elbow.to_csv(OUT/'B4_elbow.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(elbow.k,elbow.inertia,'o-')
    ax.axvline(3,color='black',ls='--',lw=1,label='chosen k=3')
    ax.set(xlabel='Number of clusters k',ylabel='Inertia (squared scaled distance)',
           title='B4: Elbow method')
    ax.set_xticks(elbow.k);ax.legend()
    save('B4_elbow.png')
    # Choose k using the elbow, without looking at cultivar labels.
    chosen_k=3
    km=KMeans(n_clusters=chosen_k,init='k-means++',n_init=10,
              max_iter=300,random_state=0).fit(Z)
    cluster_labels=km.labels_


    centres=pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),columns=X.columns)
    centres.insert(0,'cluster',range(chosen_k))
    centres.to_csv(OUT/'B5_centres_original_units.csv',index=False)
    sizes=pd.DataFrame({'cluster':range(chosen_k),
                        'size':np.bincount(cluster_labels,minlength=chosen_k)})
    sizes.to_csv(OUT/'B5_cluster_sizes.csv',index=False)
    descriptions=[describe_cluster(km.cluster_centers_[i],list(X.columns))
                  for i in range(chosen_k)]
    fig,ax=plt.subplots(figsize=(7,5))
    scatter=ax.scatter(Z_2d[:,0],Z_2d[:,1],c=cluster_labels,s=26,alpha=.75)
    ax.set(xlabel='PC1',ylabel='PC2',title='B5: PCA projection coloured by K-Means cluster')
    ax.legend(*scatter.legend_elements(),title='Cluster')
    save('B5_cluster_PCA.png')
    # Reveal the real cultivars only for evaluation.
    agreement_rows=[]
    crosstabs={}
    variants=[('scaled (13 features)',Z),('raw/unscaled (13 features)',X.to_numpy()),
              ('scaled PC1+PC2 only',Z_2d)]
    for name,features in variants:
        fitted=KMeans(n_clusters=3,init='k-means++',n_init=10,
                      max_iter=300,random_state=0).fit(features)
        score,cross=cluster_agreement(cultivar,fitted.labels_)
        agreement_rows.append({'model':name,'agreement':score})
        crosstabs[name]=cross
        print('B6',name,'agreement',score,'\n',cross.to_string())
    agreement=pd.DataFrame(agreement_rows)
    agreement.to_csv(OUT/'B_agreement.csv',index=False)
    crosstabs['scaled (13 features)'].to_csv(OUT/'B6_crosstab_scaled.csv')
    crosstabs['raw/unscaled (13 features)'].to_csv(OUT/'B6_crosstab_raw.csv')
    crosstabs['scaled PC1+PC2 only'].to_csv(OUT/'B6_crosstab_PCA2.csv')

    # Bonus: in supervised learning, fit scalers using training data only.
    xtr,xte,ytr,yte=train_test_split(X,cultivar,test_size=.25,
                            random_state=42,stratify=cultivar)
    supervised=[]
    for name,model in [
        ('Scaled KNN(5): all 13 features',
         make_pipeline(StandardScaler(),KNeighborsClassifier(5))),
        ('Scaled PCA(2) + KNN(5)',
         make_pipeline(StandardScaler(),PCA(n_components=2),KNeighborsClassifier(5)))]:
        model.fit(xtr,ytr)
        supervised.append({'model':name,'test_accuracy':accuracy_score(yte,model.predict(xte))})
    supervised=pd.DataFrame(supervised)
    supervised.to_csv(OUT/'B_bonus_supervised.csv',index=False)

    high_std=deviations.sort_values(ascending=False)
    low_std=deviations.sort_values()
    raw_range=ranges.sort_values(ascending=False)
    agreement_winner=agreement.sort_values('agreement',ascending=False).iloc[0]
    scaled_agreement=agreement.loc[agreement.model=='scaled (13 features)','agreement'].iloc[0]
    raw_agreement=agreement.loc[agreement.model=='raw/unscaled (13 features)','agreement'].iloc[0]
    two_pc_agreement=agreement.loc[agreement.model=='scaled PC1+PC2 only','agreement'].iloc[0]
    positive_pc1=pc1_sorted.tail(3).iloc[::-1].feature.tolist()
    negative_pc1=pc1_sorted.head(3).feature.tolist()
    positive_pc2=pc2_sorted.tail(3).iloc[::-1].feature.tolist()
    negative_pc2=pc2_sorted.head(3).feature.tolist()
    centre_paragraph='\n'.join(f'- **Cluster {i} ({sizes.iloc[i]["size"]} wines):** {desc}.'
                                for i,desc in enumerate(descriptions))
    two_pc_interpretation = (
        'Here the first two PCs achieve exactly the same cultivar agreement as all 13 scaled features: this suggests the key separation among these three groups is largely visible in the two-dimensional plane, even though it preserves only part of the total variance.'
        if np.isclose(two_pc_agreement,scaled_agreement) else
        'The two-PC agreement differs from the full scaled-feature agreement, because removing later principal components can also remove information that separates cultivars.'
    )
    report=f"""### B1 — Wine dataset

`load_wine(as_frame=True)` provides **{len(X)} wines** and **{X.shape[1]} chemical measurements** from three cultivars in one Italian region. The features include alcohol, acids, magnesium, phenols, colour measurements and proline. Cultivar labels are kept aside until B6. I printed `wine.DESCR` and `X.describe()`. **{', '.join(high_std.head(3).index)}** have the largest standard deviations; **{', '.join(low_std.head(3).index)}** have the smallest. **{', '.join(raw_range.head(3).index)}** have the largest ranges.

### B2 — PCA on raw data

Raw PC1 explains **{pca_raw.explained_variance_ratio_[0]:.2%}** of the variance. Its loadings, sorted by value, are:

{table(raw_loadings)}

**{raw_dominant}** dominates because its numerical scale is much larger. PCA centres data but does not automatically scale it, so this PC1 is not a balanced summary of all chemical measurements.

### B3 — PCA after scaling

After `StandardScaler`, the feature means are approximately 0 and standard deviations approximately 1.

{table(variance_table)}

**{components_90} components** explain at least 90% of the variance. PC1 and PC2 together explain **{pc2_var:.2%}**. Loadings are shown below, sorted separately by each component.

**Sorted by PC1:**

{table(pc1_sorted)}

**Sorted by PC2:**

{table(pc2_sorted)}

Positive PC1 means more **{', '.join(positive_pc1)}** and less **{', '.join(negative_pc1)}**; negative PC1 is the reverse. Positive PC2 means more **{', '.join(positive_pc2)}** and less **{', '.join(negative_pc2)}**; negative PC2 is the reverse. From the unlabelled plot, I would guess about three overlapping groups.

![Scree plot](results/B3_scree.png)
![Unlabelled PCA](results/B3_unlabelled_PCA.png)

### B4 — K-Means

With `k=3`, eight runs using `n_init=1` and seeds 0–7 gave different inertia values:

{table(seed_table)}

Different starting centres can produce different solutions. `n_init=10` tests ten starts and keeps the lowest inertia. I then calculated inertia for k=1–10:

{table(elbow)}

I chose **k={chosen_k}** near the elbow. Inertia keeps decreasing as k increases, so the minimum alone cannot determine a useful k.

![Elbow curve](results/B4_elbow.png)

### B5 — Cluster interpretation

Cluster sizes:

{table(sizes)}

Centres converted back to original measurement units:

{table(centres)}

Using the chemical measurements that differ most from the dataset average:

{centre_paragraph}

![Clusters in the PC1–PC2 plane](results/B5_cluster_PCA.png)

### B6 — Compare with cultivars

Only after clustering did I compare the groups with the actual cultivars. Cluster numbers are arbitrary.

{table(crosstabs['scaled (13 features)'].reset_index())}

Agreement is the sum of the most common cultivar count in each cluster divided by 178.

{table(agreement)}

Scaling improves agreement from **{raw_agreement:.2%}** to **{scaled_agreement:.2%}** because Euclidean distances otherwise favour large-number measurements. The first two scaled PCs retain {pc2_var:.2%} of variance yet give **{two_pc_agreement:.2%}** agreement, suggesting useful cultivar separation is visible in this plane. Agreement measures the discovered groups on these wines, not accuracy on unseen wines.

### Session B bonus — With labels

I compared two supervised KNN pipelines on a stratified 75/25 split. Scaling and PCA were fitted only on the training subset.

{table(supervised)}

These test accuracies cannot be directly compared with unsupervised agreement: clustering used all 178 wines and the cultivars were only matched afterward.
"""
    update_readme(report)
    print('B2 raw PC1:',pca_raw.explained_variance_ratio_[0])
    print('B3 90% PCs:',components_90,'first 2 PCs:',pc2_var)
    print('B4 seeds:\n',seed_table.to_string(index=False))
    print('B5 cluster sizes:\n',sizes.to_string(index=False))
    print('B6 agreements:\n',agreement.to_string(index=False))
    print('B bonus:\n',supervised.to_string(index=False))
    print('Updated README and saved results/B_agreement.csv')

if __name__=='__main__':
    main()
