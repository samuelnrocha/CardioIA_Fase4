import React, { useEffect, useRef } from "react";
import { View, Text, Animated, StyleSheet } from "react-native";
import { COLORS } from "../theme/colors";

export default function ResultCard({ result }) {
  const animWidth = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (result) {
      Animated.timing(animWidth, {
        toValue: result.confianca_pct || 0,
        duration: 550,
        useNativeDriver: false,
      }).start();
    }
  }, [result]);

  if (!result) {
    return null;
  }

  const isCritical = result.classe_id === 1 || result.status_alerta === "CRITICO";
  const accentColor = isCritical ? COLORS.fiapPink : COLORS.emerald;

  return (
    <View style={[styles.card, { borderColor: accentColor }]}>
      <View style={styles.topRow}>
        <Text style={[styles.diagTitle, { color: accentColor }]}>
          {isCritical ? "CARDIOMEGALIA DETECTADA" : "SILHUETA CARDÍACA NORMAL"}
        </Text>
        <View style={[styles.badge, { backgroundColor: accentColor }]}>
          <Text style={styles.badgeText}>{result.status_alerta}</Text>
        </View>
      </View>

      <Text style={styles.modelLabel}>
        Modelo: {result.modelo_utilizado || "ResNet18 Fine-Tuned"} • Modo: {result.modo_execucao}
      </Text>

      <View style={styles.progressTrack}>
        <Animated.View
          style={[
            styles.progressBar,
            {
              backgroundColor: accentColor,
              width: animWidth.interpolate({
                inputRange: [0, 100],
                outputRange: ["0%", "100%"],
              }),
            },
          ]}
        />
      </View>

      <View style={styles.metricsRow}>
        <View style={styles.metricBox}>
          <Text style={styles.metricLabel}>Confiança CNN</Text>
          <Text style={[styles.metricValue, { color: accentColor }]}>
            {result.confianca_pct}%
          </Text>
        </View>
        <View style={styles.metricBox}>
          <Text style={styles.metricLabel}>ICT Estimado</Text>
          <Text style={styles.metricValue}>{result.ict_estimado}</Text>
        </View>
        <View style={styles.metricBox}>
          <Text style={styles.metricLabel}>Latência</Text>
          <Text style={styles.metricValue}>{result.tempo_inferencia_ms} ms</Text>
        </View>
      </View>

      <View style={styles.multimodalBox}>
        <Text style={styles.multiHeader}>
          Fusão Multimodal (IoT ESP32 Fase 3 + Visão Computacional Fase 4)
        </Text>
        <Text style={styles.multiScore}>
          Escore Global: {result.score_multimodal} / 100 ({result.classificacao_risco_global})
        </Text>
      </View>

      <Text style={styles.recHeader}>Conduta Médica Sugerida pela IA:</Text>
      {(result.recomendacoes || [result.recomendacao_clinica]).map((item, idx) => (
        <Text key={idx} style={styles.recItem}>
          • {item}
        </Text>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    borderWidth: 1.5,
    padding: 15,
    marginTop: 12,
  },
  topRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  diagTitle: {
    fontSize: 16,
    fontWeight: "800",
    flex: 1,
  },
  badge: {
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderRadius: 99,
  },
  badgeText: {
    color: COLORS.white,
    fontWeight: "800",
    fontSize: 10.5,
  },
  modelLabel: {
    color: COLORS.textSecondary,
    fontSize: 11,
    marginTop: 4,
    marginBottom: 8,
  },
  progressTrack: {
    height: 12,
    backgroundColor: COLORS.surface,
    borderRadius: 99,
    overflow: "hidden",
    marginVertical: 8,
  },
  progressBar: {
    height: "100%",
  },
  metricsRow: {
    flexDirection: "row",
    gap: 8,
    marginVertical: 8,
  },
  metricBox: {
    flex: 1,
    backgroundColor: COLORS.surface,
    padding: 9,
    borderRadius: 8,
    alignItems: "center",
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  metricLabel: {
    color: COLORS.textSecondary,
    fontSize: 10.5,
  },
  metricValue: {
    color: COLORS.skyBlue,
    fontSize: 15,
    fontWeight: "800",
    marginTop: 2,
  },
  multimodalBox: {
    backgroundColor: "#090D16",
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 9,
    padding: 10,
    marginVertical: 8,
  },
  multiHeader: {
    color: COLORS.skyBlue,
    fontSize: 11,
    fontWeight: "700",
  },
  multiScore: {
    color: COLORS.textPrimary,
    fontSize: 12.5,
    fontWeight: "700",
    marginTop: 3,
  },
  recHeader: {
    color: COLORS.textPrimary,
    fontSize: 13,
    fontWeight: "700",
    marginTop: 6,
    marginBottom: 4,
  },
  recItem: {
    color: "#CBD5E1",
    fontSize: 12,
    marginBottom: 3,
    lineHeight: 17,
  },
});
