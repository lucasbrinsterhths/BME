import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm

# Gernerate sample data for 2 conditions
df = pd.DataFrame({
    'Dominant Hand': [305.251, 372.255, 284.750, 251.149, 145.106, 231.745, 209.681, 349.667, 225.683, 178.840, 279.910, 385.000, 260.508, 269.123, 274.638],
    'Non-dominant Hand': [279.841, 330.233, 250.607, 245.581, 116.179, 169.253, 197.663, 253.309, 150.380, 180.950, 265.509, 334.000, 265.306, 233.127, 285.963],
    'Gender': ['M', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'M', 'F', 'M', 'M'],
    'Height': [63, 72, 70, 69, 63, 62, 63, 70, 60, 67, 61, 73, 62, 66, 75]
})

# Add subject ID for repeated measures
df['Subject'] = range(1, len(df) + 1)

# Melt the dataframe for plotting and ANOVA
df_melted = df.melt(id_vars=['Gender', 'Subject'], value_vars=['Dominant Hand', 'Non-dominant Hand'], var_name='Hand', value_name='Force')

# Add Height to melted dataframe
df_melted = df_melted.merge(df[['Subject', 'Height']]
                            , on='Subject', how='left')

# t-test
t, p = stats.ttest_rel(df['Dominant Hand'], df['Non-dominant Hand'])
print(f"Paired t-test: t={t:.3g}, p={p:.3g}")

# Repeated measures ANOVA (using mixed effects model since AnovaRM doesn't support between-subject factors)
model = mixedlm("Force ~ Hand * Gender", df_melted
                , groups=df_melted["Subject"])
result = model.fit()
print(result.summary())
print("\nFixed effects p-values (3 significant figures):")
for param, pval in zip(result.params.index, result.pvalues):
    print(f"{param}: p={pval:.3g}")

# Linear regression for Dominant Hand ~ Height
print("\nLinear regression: Dominant Hand ~ Height")
model_dom = smf.ols('Q("Dominant Hand") ~ Height', data=df)
result_dom = model_dom.fit()
print(result_dom.summary())
print(f"p-value for Height: {result_dom.pvalues['Height']:.3g}")

# Linear regression for Non-dominant Hand ~ Height
print("\nLinear regression: Non-dominant Hand ~ Height")
model_non = smf.ols('Q("Non-dominant Hand") ~ Height', data=df)
result_non = model_non.fit()
print(result_non.summary())
print(f"p-value for Height: {result_non.pvalues['Height']:.3g}")

# Model to compare effects of Gender and Height on Grip Strength
print("\nMixed effects model: Force ~ Height + Gender + Hand")
model_compare = mixedlm("Force ~ Height + Gender + Hand", df_melted, groups="Subject")
result_compare = model_compare.fit()
print(result_compare.summary())

# Calculate standardized effect sizes (beta coefficients)
sd_force = df_melted['Force'].std()
sd_height = df['Height'].std()
sd_gender = df['Gender'].map({'M': 1, 'F': 0}).std()  # SD for binary variable

beta_height = result_compare.params['Height'] * (sd_height / sd_force)
beta_gender = result_compare.params['Gender[T.M]'] * (sd_gender / sd_force)

print(f"\nStandardized effect size (β) for Height: {beta_height:.3f}")
print(f"Standardized effect size (β) for Gender: {beta_gender:.3f}")

if abs(beta_height) > abs(beta_gender):
    print("Height has a larger standardized effect on grip strength.")
else:
    print("Gender has a larger standardized effect on grip strength.")

# Box and whisker plot
sns.swarmplot(data=df_melted, x='Hand', y='Force', hue='Gender', size=10, palette="Set2", dodge=True)
sns.violinplot(data=df_melted, x='Hand', y='Force', hue='Gender', inner=None, palette="Set2", alpha=0.5, dodge=True)
plt.title('Grip Strength of Dominant and Non-dominant Hands by Gender')
plt.xlabel('Handedness')
plt.ylabel('Mean Force (N)')
plt.show()

# Regression plots for Height vs Grip Strength
fig, axes = plt.subplots(1, 2
                         , figsize=(10, 5), sharey=True)
sns.regplot(x='Height', y='Dominant Hand', data=df, ax=axes[0], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
axes[0].set_title('Dominant Hand vs Height')
axes[0].set_xlabel('Height (inches)')
axes[0].set_ylabel('Grip Strength (N)')

sns.regplot(x='Height', y='Non-dominant Hand'
            , data=df, ax=axes[1], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
axes[1].set_title('Non-dominant Hand vs Height')
axes[1].set_xlabel('Height (inches)')
axes[1].set_ylabel('Grip Strength (N)')

plt.tight_layout()
plt.show()
