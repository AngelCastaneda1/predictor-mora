# ============================================================
# PREDICCIÓN DE RIESGO CREDITICIO - Give Me Some Credit
# Persona 1: Datos, Modelo Logit y Evaluación
# ============================================================

# ---- 1. LIBRERÍAS ----
library(tidyverse)
library(caret)
library(stargazer)
library(pROC)
library(ggplot2)

# ---- 2. CARGAR DATOS ----
df <- read.csv("cs-training.csv")
df <- df[, -1]  # Eliminar columna índice (Unnamed: 0)

# Renombrar columnas para facilidad
colnames(df) <- c(
  "mora",                  # Variable dependiente (1 = mora, 0 = no mora)
  "uso_credito",           # Utilización de líneas de crédito
  "edad",                  # Edad del cliente
  "atraso_30_59",          # Veces atrasado 30-59 días
  "ratio_deuda",           # Ratio deuda/ingreso
  "ingreso_mensual",       # Ingreso mensual
  "creditos_abiertos",     # Número de créditos abiertos
  "atraso_90",             # Veces atrasado +90 días
  "creditos_hipotecarios", # Créditos hipotecarios
  "atraso_60_89",          # Veces atrasado 60-89 días
  "dependientes"           # Número de dependientes
)

cat("=== DIMENSIONES DE LA BASE ===\n")
cat("Filas:", nrow(df), "| Columnas:", ncol(df), "\n\n")

# ---- 3. LIMPIEZA DE DATOS ----

# Ver nulos antes de limpiar
cat("=== VALORES NULOS ANTES DE LIMPIAR ===\n")
print(colSums(is.na(df)))

# Imputar nulos con la mediana (más robusta que la media)
df$ingreso_mensual[is.na(df$ingreso_mensual)] <- median(df$ingreso_mensual, na.rm = TRUE)
df$dependientes[is.na(df$dependientes)]       <- median(df$dependientes, na.rm = TRUE)

# Eliminar outliers extremos en uso_credito y ratio_deuda
df <- df %>% filter(uso_credito <= 1, ratio_deuda <= 1)

# Eliminar edades imposibles
df <- df %>% filter(edad >= 18 & edad <= 100)

cat("\n=== VALORES NULOS DESPUÉS DE LIMPIAR ===\n")
print(colSums(is.na(df)))
cat("\nFilas después de limpieza:", nrow(df), "\n\n")

# ---- 4. ESTADÍSTICAS DESCRIPTIVAS ----
cat("=== TABLA DESCRIPTIVA ===\n")
desc <- df %>%
  summarise(across(everything(), list(
    media   = ~round(mean(.), 3),
    mediana = ~round(median(.), 3),
    sd      = ~round(sd(.), 3),
    min     = ~round(min(.), 3),
    max     = ~round(max(.), 3)
  )))
print(as.data.frame(t(desc)))

# Distribución de la variable dependiente
cat("\n=== DISTRIBUCIÓN DE MORA ===\n")
tabla_mora <- table(df$mora)
print(tabla_mora)
cat("% que cayó en mora:", round(prop.table(tabla_mora)[2] * 100, 2), "%\n\n")

# ---- 5. GRÁFICO: DISTRIBUCIÓN DE MORA ----
ggplot(df, aes(x = factor(mora), fill = factor(mora))) +
  geom_bar() +
  scale_fill_manual(values = c("#2196F3", "#F44336"),
                    labels = c("No mora (0)", "Mora (1)")) +
  labs(title = "Distribución de la Variable Dependiente (Mora)",
       x = "Mora", y = "Número de clientes", fill = "") +
  theme_minimal()
ggsave("grafico_distribucion_mora.png", width = 7, height = 5)

# ---- 6. DIVISIÓN TRAIN / TEST ----
set.seed(123)
indice <- createDataPartition(df$mora, p = 0.75, list = FALSE)
train  <- df[indice, ]
test   <- df[-indice, ]

cat("=== DIVISIÓN TRAIN/TEST ===\n")
cat("Entrenamiento:", nrow(train), "filas\n")
cat("Prueba:", nrow(test), "filas\n\n")

# ---- 7. MODELO LOGIT ----
modelo_logit <- glm(mora ~ uso_credito + edad + atraso_30_59 + ratio_deuda +
                      ingreso_mensual + creditos_abiertos + atraso_90 +
                      creditos_hipotecarios + atraso_60_89 + dependientes,
                    data   = train,
                    family = binomial(link = "logit"))

cat("=== RESUMEN DEL MODELO LOGIT ===\n")
summary(modelo_logit)

# Exportar tabla de regresión con stargazer
stargazer(modelo_logit,
          type  = "text",
          title = "Modelo Logit - Predicción de Mora",
          out   = "tabla_regresion.txt")

# ---- 8. INTERPRETACIÓN DE COEFICIENTES ----
cat("\n=== ODDS RATIOS (exp(coeficientes)) ===\n")
odds <- exp(coef(modelo_logit))
print(round(odds, 4))

cat("\n--- INTERPRETACIÓN ---\n")
cat("uso_credito: Por cada unidad adicional en uso de crédito,\n")
cat("  el odds de caer en mora se multiplica por", round(odds["uso_credito"], 4), "\n\n")

cat("atraso_90: Por cada vez adicional con atraso >90 días,\n")
cat("  el odds de caer en mora se multiplica por", round(odds["atraso_90"], 4), "\n\n")

cat("edad: Por cada año adicional de edad,\n")
cat("  el odds de caer en mora se multiplica por", round(odds["edad"], 4), "\n")

# ---- 9. PREDICCIONES ----
prob_pred <- predict(modelo_logit, newdata = test, type = "response")
clase_pred <- ifelse(prob_pred >= 0.5, 1, 0)

# ---- 10. MATRIZ DE CONFUSIÓN ----
cat("\n=== MATRIZ DE CONFUSIÓN ===\n")
conf_matrix <- confusionMatrix(
  factor(clase_pred),
  factor(test$mora),
  positive = "1"
)
print(conf_matrix)

# Guardar métricas clave
accuracy  <- round(conf_matrix$overall["Accuracy"] * 100, 2)
precision <- round(conf_matrix$byClass["Precision"] * 100, 2)
recall    <- round(conf_matrix$byClass["Recall"] * 100, 2)

cat("\n--- MÉTRICAS CLAVE ---\n")
cat("Accuracy (exactitud):", accuracy, "%\n")
cat("Precision:", precision, "%\n")
cat("Recall (sensibilidad):", recall, "%\n")

# ---- 11. CURVA ROC y AUC ----
roc_obj <- roc(test$mora, prob_pred)
auc_val <- round(auc(roc_obj), 4)
cat("\nAUC (Área bajo la curva ROC):", auc_val, "\n")

png("grafico_roc.png", width = 700, height = 500)
plot(roc_obj,
     main = paste("Curva ROC - AUC =", auc_val),
     col  = "#2196F3", lwd = 2)
dev.off()

# ---- 12. GRÁFICO DE ERRORES (Predicho vs Real) ----
resultados <- data.frame(
  real      = test$mora,
  predicho  = clase_pred,
  prob      = prob_pred
)

ggplot(resultados, aes(x = prob, fill = factor(real))) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("#2196F3", "#F44336"),
                    labels = c("No mora", "Mora")) +
  labs(title = "Distribución de Probabilidades Predichas vs. Valor Real",
       x = "Probabilidad de mora predicha", y = "Frecuencia", fill = "Real") +
  theme_minimal()
ggsave("grafico_errores.png", width = 8, height = 5)

# ---- 13. GUARDAR MODELO ----
saveRDS(modelo_logit, "modelo_logit_mora.rds")
cat("\n✅ Modelo guardado como 'modelo_logit_mora.rds'\n")
cat("✅ Tablas y gráficos exportados correctamente\n")
cat("\n--- FIN DEL SCRIPT ---\n")
