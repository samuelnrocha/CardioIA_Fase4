import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { COLORS } from "../theme/colors";

export default function Header({ mockMode }) {
  return (
    <View style={styles.header}>
      <View style={styles.row}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>FIAP • FASE 4</Text>
        </View>
        <View
          style={[
            styles.modeBadge,
            { borderColor: mockMode ? COLORS.amber : COLORS.emerald },
          ]}
        >
          <Text
            style={[
              styles.modeText,
              { color: mockMode ? COLORS.amber : COLORS.emerald },
            ]}
          >
            {mockMode ? "MODO DEMO OFFLINE" : "API FLASK CONECTADA"}
          </Text>
        </View>
      </View>
      <Text style={styles.title}>CardioIA — Visão Computacional</Text>
      <Text style={styles.subtitle}>
        Data Health Solutions • Assistente Cardiológico Virtual (ResNet18 + IoT)
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    backgroundColor: "#090D16",
    borderBottomWidth: 2,
    borderBottomColor: COLORS.fiapPink,
    paddingHorizontal: 18,
    paddingTop: 18,
    paddingBottom: 14,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  badge: {
    backgroundColor: COLORS.fiapPink,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 5,
  },
  badgeText: {
    color: COLORS.white,
    fontWeight: "800",
    fontSize: 11,
  },
  modeBadge: {
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 99,
  },
  modeText: {
    fontSize: 10,
    fontWeight: "700",
  },
  title: {
    color: COLORS.textPrimary,
    fontSize: 19,
    fontWeight: "800",
  },
  subtitle: {
    color: COLORS.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
});
