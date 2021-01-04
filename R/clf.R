library(reticulate)

#reticulate::source_python("classify_functions.py")

#df <- data.table::fread('bbc_data.csv')

# foo <- prep_data("bbc_data.csv")
# 
# category <- unique(unlist(foo[[3]][3]) )
# 
# 
# 
# x <- log_reg(foo[[1]] , foo[[2]], t_size=0.33)
# y <- knn(foo[[1]] , foo[[2]], t_size=0.33)
# z <- svm_clf(foo[[1]] , foo[[2]], t_size=0.33)
# 
# a <- gnb_clf(foo[[1]] , foo[[2]], t_size=0.33)
# b <- dtrees_clf(foo[[1]] , foo[[2]], t_size=0.33)
# c <- rf_clf(foo[[1]] , foo[[2]], t_size=0.33)
# 
# conf_matrix <- as.data.frame(x[[1]] , row.names= category)
# colnames(conf_matrix) <- category
# 
# x[[3]]
# y[[3]]
# z[[3]]
# a[[3]]
# b[[3]]
# c[[1]]