import React from "react";
import { View, Text, TouchableOpacity, Switch, StyleSheet } from "react-native";
import { COLORS } from "../theme/colors";

export default function HomeScreen({
  mockMode,
  onToggleMock,
  telemetry,
  onNavigateDiagnostic,
}) {
  return (
    <View style={styles.container}>
      <View style={styles.heroCard}>
        <Text style={styles.heroTag}>ECOSSISTEMA MULTIMODAL CARDIOIA</Text>
        <Text style={styles.heroTitle}>
          Triagem Inteligente de Cardiomegalia por Raio-X e Sinais Vitais IoT
        </Text>
        <Text style={styles.heroDesc}>
          Integra o modelo convolucional ResNet18 Fine-Tuned (ROC-AUC 99%+) com a
          telemetria contínua do vestível ESP32 da Fase 3.
        </Text>

        <View style={styles.switchRow}>
          <View>
            <Text style={styles.switchTitle}>Modo Demonstração Offline (Mock Mode)</Text>
            <Text style={styles.switchSub}>
              Permite testar a inferência mesmo sem servidor Flask ativo na rede
            </Text>
          </View>
          <Switch
            value={mockMode}
            onValueChange={onToggleMock}
            trackColor={{ false: "#334155", true: COLORS.fiapPink }}
          />
        </View>
      </View>

      <View style={styles.telemetryCard}>
        <Text style={styles.sectionTitle}>
          📡 Telemetria em Tempo Real (Vestível ESP32 — Fase 3)
        </Text>
        <View style={styles.vitalRow}>
          <View style={styles.vitalBox}>
            <Text style={styles.vitalLabel}>Batimentos</Text>
            <Text style={[styles.vitalVal, { color: COLORS.fiapPink }]}>
              {telemetry.bpm} BPM
            </Text>
          </View>
          <View style={styles.vitalBox}>
            <Text style={styles.vitalLabel}>Temperatura</Text>
            <Text style={[styles.vitalVal, { color: COLORS.amber }]}>
              {telemetry.temperatura} °C
            </Text>
          </View>
          <View style={styles.vitalBox}>
            <Text style={styles.vitalLabel}>MPU6050</Text>
            <Text style={[styles.vitalVal, { color: COLORS.emerald }]}>
              {telemetry.queda_detectada ? "ALERTA QUEDA" : "ATIVO"}
            </Text>
          </View>
        </View>
      </View>

      <TouchableOpacity style={styles.primaryBtn} onPress={onNavigateDiagnostic}>
        <Text style={styles.primaryBtnText}>
          Iniciar Análise de Exame Radiológico (CNN) →
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  heroCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 16,
    marginBottom: 14,
  },
  heroTag: {
    color: COLORS.fiapPink,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  heroTitle: {
    color: COLORS.textPrimary,
    fontSize: 18,
    fontWeight: "800",
    marginTop: 5,
    marginBottom: 6,
  },
  heroDesc: {
    color: COLORS.textSecondary,
    fontSize: 13,
    lineHeight: 19,
  },
  switchRow: {
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  switchTitle: {
    color: COLORS.textPrimary,
    fontSize: 13,
    fontWeight: "700",
  },
  switchSub: {
    color: COLORS.textSecondary,
    fontSize: 11,
  },
  telemetryCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 15,
    marginBottom: 16,
  },
  sectionTitle: {
    color: COLORS.skyBlue,
    fontSize: 13.5,
    fontWeight: "700",
    marginBottom: 10,
  },
  vitalRow: {
    flexDirection: "row",
    gap: 8,
  },
  vitalBox: {
    flex: 1,
    backgroundColor: COLORS.surface,
    padding: 10,
    borderRadius: 9,
    alignItems: "center",
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  vitalLabel: {
    color: COLORS.textSecondary,
    fontSize: 11,
  },
  vitalVal: {
    fontSize: 15,
    fontWeight: "800",
    marginTop: 3,
  },
  primaryBtn: {
    backgroundColor: COLORS.fiapPink,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  primaryBtnText: {
    color: COLORS.white,
    fontSize: 14.5,
    fontWeight: "800",
  },
});
