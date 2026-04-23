##################
## Read in parameters from py
##################

args <- commandArgs(trailingOnly = TRUE)

print("BLAS/OpenMP thread settings seen by R:")
print(Sys.getenv(c(
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "BLAS_NUM_THREADS"
)))

input_root <- args[1]
prefix <- args[2]
ncores <- as.integer(args[3])
k_num <- as.integer(args[4])
if_fixThis <- args[5]
if_fixThis <- ifelse(if_fixThis == "True", TRUE, FALSE)
data_type <- args[6]
family <- ifelse(data_type == "count", "pois", "gaussian")
out_root <- args[7]

input_dir <- file.path(input_root, paste0(prefix, "_gam_input"))
out_dir <- file.path(out_root, paste0(prefix, "_gam_output"))

# Print values to R console
cat("---- Arguments input for GAM smothing in R subprocess----\n")
cat("prefix:     ", prefix,     "\n")
cat("input_dir:  ", input_dir,  "\n")
cat("out_dir:    ", out_dir,    "\n")
cat("family:     ", family,     "\n")
cat("k_num:      ", k_num,      "\n")
cat("ncores:     ", ncores,     "\n")
cat("---------------------\n")

###################
## load in raw gene expression
###################
suppressPackageStartupMessages({
  library(mgcv)
  library(Matrix)
  library(data.table)
  library(dplyr)
  library(metapod)
})

# Construct file paths
mtx_path <- file.path(input_dir, paste0(prefix, "_counts.mtx"))
genes_path <- file.path(input_dir, paste0(prefix, "_genes.tsv"))
barcodes_path <- file.path(input_dir, paste0(prefix, "_barcodes.tsv"))
obs_path <- file.path(input_dir, paste0(prefix, "_obs_metadata.csv"))

# Load inputs
countMat <- Matrix::readMM(mtx_path)
genes <- readLines(genes_path)
barcodes <- readLines(barcodes_path)
colData_df <- read.csv(obs_path)
rownames(colData_df) <- colData_df$barcode

# Assign dimnames to countMat
rownames(countMat) <- barcodes
colnames(countMat) <- genes
countMat <- t(countMat)

## filter lowly expressed genes 
print("Lowly expressed genes (<1 pct across all spots) are being removed...")
selected_genes <- names(which(Matrix::rowMeans( countMat >0 ) > 0.01))
selected_countMat <- countMat[selected_genes,]

selected_colData_df <- colData_df
if ("sum_umi" %in% colnames(selected_colData_df)) {
  selected_colData_df$log_sum_umi <- log(selected_colData_df$sum_umi)
}
print(paste0("This numbe of genes are being smoothed: ", nrow(selected_countMat)))
rm(countMat)
rm(colData_df)

#################################
## gene smoothing and SVG testing
#################################
test_pred_gam <- function(This_gene_name,
                          This_gene_exp,
                          spot_meta,
                          num_of_basis = 40,
                          if_fx = TRUE,
                          dist = "pois"){
  
  ## testing gam sample by sample
  samples_tobe_test <- unique(spot_meta$sample_name)
  stat_outs <- list()
  pred_outs <- list()
  for(x in 1:length(samples_tobe_test)){
    This_sample_name <- samples_tobe_test[x]
    
    ## separate samples
    col_index <- which(spot_meta$sample_name == This_sample_name)
    This_sample_spot_meta <- spot_meta[col_index,]
    This_sample_gene_exp <- This_gene_exp[This_sample_spot_meta$barcode]
    
    This_sample_SVG_input <- data.frame(Y = This_sample_gene_exp,This_sample_spot_meta)
    
    ## testing for gene - pois distribution 
    if(dist == "pois"){
      if(if_fx == TRUE){
        full_mod <- mgcv::gam(Y~s(row,col,
                                  fx=TRUE,
                                  k=num_of_basis), 
                              family = poisson(), 
                              offset = log_sum_umi,
                              data = This_sample_SVG_input)
      }else{
        full_mod <- mgcv::gam(Y~s(row,col,
                                  fx=FALSE,
                                  k=num_of_basis), 
                              family = poisson(), 
                              offset = log_sum_umi,
                              data = This_sample_SVG_input)
      }
      
      full_mod_sum <- summary(full_mod)
      base_mod <- mgcv::gam(Y~1, family = poisson(), 
                            offset = log_sum_umi,
                            data = This_sample_SVG_input)
      base_mod_sum <- summary(base_mod)
      
    }
    
    if(dist == "gaussian"){
      if(if_fx == TRUE){
        full_mod <- mgcv::gam(Y~s(row,col,
                                  fx=TRUE,
                                  k=num_of_basis), 
                              data = This_sample_SVG_input)
      }else{
        full_mod <- mgcv::gam(Y~s(row,col,
                                  fx=FALSE,
                                  k=num_of_basis), 
                              data = This_sample_SVG_input)
      }
      
      full_mod_sum <- summary(full_mod)
      base_mod <- mgcv::gam(Y~1,
                            data = This_sample_SVG_input)
      base_mod_sum <- summary(base_mod)
      
    }
    
    
    test_rst <- anova(full_mod, base_mod, test = "Chisq")
    
    This_sample_stat <- data.frame(sample_name = This_sample_name,
                                   genes = This_gene_name,
                                   pval = test_rst$`Pr(>Chi)`[2],
                                   full_mod_edf = full_mod_sum$edf,
                                   full_mod_Ref.df= full_mod_sum$s.table[2],
                                   full_mod_Intercept = full_mod_sum$p.coeff[1],
                                   base_mod_Intercept = base_mod_sum$p.coeff[1],
                                   dist = dist,
                                   if_fx = if_fx)
    row.names(This_sample_stat) <- NULL
    stat_outs[[x]] <- This_sample_stat
    
    if(dist == "pois"){
      This_sample_pred <- log(exp(full_mod$linear.predictors - full_mod$offset)+1)
    }
    
    if(dist == "gaussian"){
      This_sample_pred <- full_mod$linear.predictors
    }
    pred_outs[[x]] <- This_sample_pred
  }
  
  stat_outs <- do.call(rbind,stat_outs)
  pred_outs <- unlist(pred_outs)
  
  ## if a gene is not shown sig in any sample, the we dont record the pred exp
  if(sum(stat_outs$pval<0.05) ==0){
    pred_outs <- NA
  }
  
  return(list(This_gene_stat = stat_outs,
              This_gene_pred = pred_outs))
}



######### testing gene by gene
print("Start spatial smoothing by GAM...")
outs <- parallel::mclapply(1:nrow(selected_countMat), function(i){
  
  tryCatch({
    one_gene_outs <- test_pred_gam(This_gene_name = rownames(selected_countMat)[i],
                                   This_gene_exp = selected_countMat[i,],
                                   spot_meta = selected_colData_df,
                                   num_of_basis = k_num,
                                   if_fx = if_fixThis,
                                   dist = family)
    
    return(list(This_gene_stat = one_gene_outs$This_gene_stat,
                This_gene_pred = one_gene_outs$This_gene_pred,
                gene_names =  rownames(selected_countMat)[i]))
  },
  error=function(e){ })
  
  
  
},
mc.cores = ncores)

print("GAM done!")

print("Preparing GAM test statistics outs...")
final_stat_outs <-  lapply(outs, function(x){
  return(x$This_gene_stat)
})
final_stat_outs <- do.call(rbind,final_stat_outs) # GAM test statistics in the long format 

print("Preparing smoothed gene expression outs...")
final_pred_outs <-  lapply(outs, function(x){
  
  return(x$This_gene_pred)
})
final_pred_outs <- do.call(rbind, final_pred_outs)
gene_names <- unlist(lapply(outs, function(x){
  
  return(x$gene_names)
}))
rownames(final_pred_outs) <- gene_names
colnames(final_pred_outs) <- selected_colData_df$barcode
final_pred_outs <- na.omit(final_pred_outs) # GAM smoothed gene expression


print(paste0(nrow(selected_countMat) - nrow(final_pred_outs), " genes are removed due to being not spatially variable in any samples..."))
rm(outs)

######################################
## test statistics summary and outting
#######################################
write_matrix <-function(temp_pred_outs, ## the final output smoothed gene expression (matrix)
                        temp_test_stat_combined, ## the gam testing stats cmobined (df)
                        temp_test_stat, ##  the gam testing stats per sample (df)
                        spot_meta, ## meta data for the obs (df)
                        out_dir, ## the directory for storing the intermediate results
                        prefix ## the prefix for the intermediate results 
){
  
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  
  # Check that rownames and colnames exist
  stopifnot(!is.null(rownames(temp_pred_outs)))
  stopifnot(!is.null(colnames(temp_pred_outs)))
  
  # convert gene expression matrix to data.table
  expr_dt <- as.data.table(temp_pred_outs)
  expr_dt[, barcode := rownames(temp_pred_outs)]
  setcolorder(expr_dt, c("barcode", setdiff(names(expr_dt), "barcode")))
  fwrite(expr_dt,file.path(out_dir, paste0(prefix, "_smoothed_gene_exp.csv")),
         quote = FALSE)
  
  # Save GAM test results
  fwrite(temp_test_stat, file = file.path(out_dir, paste0(prefix, "_GAM_RawStat.csv")))
  fwrite(temp_test_stat_combined, file = file.path(out_dir, paste0(prefix, "_GAM_CombinedStat.csv")))
  
  # Save obs metadata
  fwrite(spot_meta, file = file.path(out_dir, paste0(prefix, "_spots_metadata.csv")))
}

combine_stats <- function(test_stat # GAM test stat in the long format
){
  
  # 1. Get all sample names
  all_sample_names <- unique(test_stat$sample_name)
  
  # 2. Build a named list of per-sample p-values
  test_pval_list <- lapply(all_sample_names, function(x) {
    temp <- test_stat[test_stat$sample_name == x, c("genes", "pval")]
    outs <- temp$pval
    names(outs) <- temp$genes
    return(outs)
  })
  
  # 3-0. Combine p-values using selected method (e.g., "simes", "fisher", "stouffer")
  outs_pval_combined_fisher <- metapod::combineParallelPValues(p.values = test_pval_list,
                                                               method = "fisher")
  outs_pval_combined_stouffer <- metapod::combineParallelPValues(p.values = test_pval_list,
                                                                 method = "stouffer")
  outs_pval_combined_simes <- metapod::combineParallelPValues(p.values = test_pval_list,
                                                              method = "simes")
  # 3-1.combine p-value ranks
  test_pval_list_ranks <- lapply(test_pval_list, function(x){
    rank(x,
         na.last = TRUE,
         ties.method = "min")/length(x)
  })
  test_pval_list_ranks <- do.call(rbind,test_pval_list_ranks)
  outs_pval_combined_ranks <- colSums(test_pval_list_ranks)
  outs_pval_combined_ranks <- outs_pval_combined_ranks/(max(outs_pval_combined_ranks)-min(outs_pval_combined_ranks))
  
  # 3-2.detect common SVGs
  test_pval_list_BH_corrected <- lapply(test_pval_list, function(x){
    p.adjust(x,method = "BH") <= 0.05
  })
  test_pval_list_BH_corrected <-  do.call(rbind,test_pval_list_BH_corrected)
  test_pval_list_BH_corrected <- colSums(test_pval_list_BH_corrected)/nrow(test_pval_list_BH_corrected)
  outs_pval_combined_commonSVGs <- (test_pval_list_BH_corrected == 1)
  outs_pval_combined_unionSVGs <- (test_pval_list_BH_corrected >0 )
  
  
  # 4. Make data.frame of combined p-values
  combined_pvals <- data.frame(
    genes = names(test_pval_list[[1]]),
    pval_combined_ranks = outs_pval_combined_ranks,
    pval_combined_fisher  = outs_pval_combined_fisher$p.value,
    pval_combined_stouffer  = outs_pval_combined_stouffer$p.value,
    pval_combined_simes  = outs_pval_combined_simes$p.value,
    if_common_SVGs = outs_pval_combined_commonSVGs,
    if_union_SVGs = outs_pval_combined_unionSVGs
  ) %>%
    arrange(pval_combined_ranks)
  
  # 5. Compute additional stats per gene
  combined_test_stat <- test_stat %>%
    group_by(genes) %>%
    summarize(
      n_samples   = n(),
      n_valid_p   = sum(!is.na(pval) & pval > 0 & pval < 1),
      pval_min    = min(pval, na.rm = TRUE),
      pval_max    = max(pval, na.rm = TRUE),
      pval_mean   = mean(pval, na.rm = TRUE),
      edf_mean    = mean(full_mod_edf, na.rm = TRUE),
      edf_sd      = sd(full_mod_edf, na.rm = TRUE),
      .groups     = "drop"
    ) %>%
    left_join(combined_pvals, by = "genes") %>%
    arrange(pval_combined_ranks)
  
  return(as.data.frame(combined_test_stat))
}


## summarize test statistics across samples
print("Summarizing test statistics...")
combined_test_stat <- combine_stats(final_stat_outs)

## preparing for stage 2: spatial domain detection 
print("Preparing for stage 2: spatial domain detection...")
write_matrix(final_pred_outs, ## the final output smoothed gene expression (matrix)
             combined_test_stat, ## the combined gam testing stats (df)
             final_stat_outs, ## the org gam testing stats in the long format(df)
             selected_colData_df, ## meta data for the obs (df)
             out_dir, ## the directory for storing the intermediate results
             prefix ## the prefix for the intermediate results 
)

rm(list=ls())

