from pandas_plink import read_plink
import pandas as pd

(bim, fam, bed) = read_plink('data/genotype')
geno_arr = bed.compute()
print("geno_arr.shape", geno_arr.shape)
print("len(bim.snp.values)", len(bim.snp.values))
print("len(fam.iid.values)", len(fam.iid.values))
# For samples as rows, SNPs as columns (recommended for ML)
geno_df = pd.DataFrame(geno_arr.T, index=fam.iid.values, columns=bim.snp.values)
geno_df.to_csv("data/genotype.csv")
print("CSV saved as data/genotype.csv")