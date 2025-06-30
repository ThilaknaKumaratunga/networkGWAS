import pandas as pd
from collections import defaultdict

def read_neighborhoods(neigh_file):
    """
    Reads the neighborhoods.txt file: <SNP> <Gene>
    Returns a dict: {gene: [snp1, snp2, ...]}
    """
    neighborhoods = defaultdict(list)
    with open(neigh_file, "r") as f:
        for line in f:
            snp, gene = line.strip().split()
            neighborhoods[gene].append(snp)
    return neighborhoods

def load_genotype(geno_csv):
    """
    Loads genotype data (samples x SNPs).
    Assumes columns: sample_id, snp1, snp2, ...
    """
    return pd.read_csv(geno_csv, index_col=0)

def aggregate_genotype_matrix(geno_df, neighborhoods):
    """
    Builds an aggregated genotype matrix.
    Each column = all SNPs from a gene neighborhood (flattened).
    Each row = one sample.
    Returns: DataFrame (samples x aggregated SNPs)
    """
    agg_features = []
    agg_columns = []
    for gene, snps in neighborhoods.items():
        present_snps = [snp for snp in snps if snp in geno_df.columns]
        if not present_snps:
            continue
        agg_features.append(geno_df[present_snps])
        agg_columns.extend([f"{gene}_{snp}" for snp in present_snps])
    if not agg_features:
        raise ValueError("No matching SNPs found between neighborhoods.txt and genotype.csv.")
    agg_matrix = pd.concat(agg_features, axis=1)
    agg_matrix.columns = agg_columns
    return agg_matrix

def save_matrix(matrix, output_csv):
    matrix.to_csv(output_csv)

if __name__ == "__main__":
    # Original file paths (unchanged)
    NEIGHBORHOODS_TXT = "results/settings/neighborhoods.txt"
    GENOTYPE_CSV = "data/genotype.csv"    # Your genotype data (samples x SNPs)
    OUTPUT_CSV = "cnn_ready_genotype_subset.csv"

    import pandas as pd
    from collections import defaultdict

    # --- Subset genotype ---
    print("Loading genotype data...")
    geno_df_full = pd.read_csv(GENOTYPE_CSV, index_col=0)
    print(f"Full genotype shape: {geno_df_full.shape}")
    # Use only first 100 samples and first 100 SNPs (adjust as needed)
    print("Subsetting genotype data to first 100 samples and SNPs...")
    geno_df = geno_df_full.iloc[:100, :100]
    print(f"Subset genotype shape: {geno_df.shape}")

    # --- Subset neighborhoods ---
    # Read all lines
    with open(NEIGHBORHOODS_TXT) as f:
        lines = f.readlines()

    # Keep only SNPs/genes present in the genotype subset
    snp_set = set(geno_df.columns)
    gene_order = []
    gene2snps = defaultdict(list)
    for line in lines:
        snp, gene = line.strip().split()
        if snp in snp_set:
            gene2snps[gene].append(snp)
            if gene not in gene_order:
                gene_order.append(gene)
    # Limit to first 10 genes (adjust as needed)
    subset_genes = gene_order[:10]
    neighborhoods = {g: gene2snps[g] for g in subset_genes if gene2snps[g]}

    print(f"Subset neighborhoods: {len(neighborhoods)} genes, {sum(len(snps) for snps in neighborhoods.values())} SNPs")

   

    agg_matrix = aggregate_genotype_matrix(geno_df, neighborhoods)
    agg_matrix.to_csv(OUTPUT_CSV)
    print(f"Saved CNN-ready subset matrix: {OUTPUT_CSV}")

    geno_df = pd.read_csv("cnn_ready_genotype_subset.csv", index_col=0)
    pheno_df = pd.read_csv("data/y_50.pheno", delim_whitespace=True, header=None, names=["FID", "IID", "phenotype"])
    # Use IID as index for merging
    pheno_df = pheno_df.set_index("IID")
    # Merge on index (IID), drop FID if not needed
    merged = geno_df.join(pheno_df["phenotype"], how="inner")

    # Save the merged file
    merged.to_csv("cnn_ready_with_pheno.csv")
    print("Saved merged CNN-ready file with phenotype: cnn_ready_with_pheno.csv")
