import React from "react";
import { View, Text, TouchableOpacity, Image, StyleSheet } from "react-native";
import { COLORS } from "../theme/colors";

export default function ImagePickerBox({
  selectedImage,
  sampleLabel,
  onPickCamera,
  onPickGallery,
  onSelectSample,
}) {
  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>1. Seleção de Exame Radiológico (Tórax PA/AP)</Text>

      <View style={styles.dropzone}>
        {selectedImage ? (
          <Image source={{ uri: selectedImage }} style={styles.preview} />
        ) : (
          <Text style={styles.placeholder}>
            Nenhuma radiografia selecionada. Capture pela câmera, escolha da galeria ou carregue uma amostra clínica abaixo.
          </Text>
        )}
        {sampleLabel ? <Text style={styles.sampleTag}>{sampleLabel}</Text> : null}
      </View>

      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.actionBtn} onPress={onPickCamera}>
          <Text style={styles.actionBtnText}>📷 Tirar Foto</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionBtn} onPress={onPickGallery}>
          <Text style={styles.actionBtnText}>🖼️ Galeria</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.quickLabel}>Exemplos Clínicos Pré-Carregados (1-Clique):</Text>
      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={[styles.sampleBtn, styles.sampleNormal]}
          onPress={() => onSelectSample("normal")}
        >
          <Text style={[styles.sampleBtnText, { color: COLORS.emerald }]}>
            ✓ Caso Normal (ICT 0,43)
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.sampleBtn, styles.sampleCardio]}
          onPress={() => onSelectSample("cardiomegalia")}
        >
          <Text style={[styles.sampleBtnText, { color: COLORS.neonPink }]}>
            🚨 Cardiomegalia (ICT 0,58)
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 15,
    marginBottom: 14,
  },
  sectionTitle: {
    color: COLORS.textPrimary,
    fontSize: 15,
    fontWeight: "700",
    marginBottom: 10,
  },
  dropzone: {
    borderWidth: 1.5,
    borderStyle: "dashed",
    borderColor: COLORS.fiapPink,
    borderRadius: 12,
    padding: 14,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 150,
    backgroundColor: "rgba(237, 20, 91, 0.05)",
    marginBottom: 12,
  },
  preview: {
    width: 165,
    height: 165,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.skyBlue,
  },
  placeholder: {
    color: COLORS.textSecondary,
    textAlign: "center",
    fontSize: 12.5,
  },
  sampleTag: {
    marginTop: 8,
    color: COLORS.skyBlue,
    fontSize: 11.5,
    fontWeight: "600",
  },
  buttonRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 10,
  },
  actionBtn: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: "center",
  },
  actionBtnText: {
    color: COLORS.textPrimary,
    fontWeight: "600",
    fontSize: 13,
  },
  quickLabel: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    marginBottom: 6,
  },
  sampleBtn: {
    flex: 1,
    paddingVertical: 9,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: "center",
  },
  sampleNormal: {
    borderColor: COLORS.emerald,
    backgroundColor: "rgba(16, 185, 129, 0.12)",
  },
  sampleCardio: {
    borderColor: COLORS.fiapPink,
    backgroundColor: "rgba(237, 20, 91, 0.14)",
  },
  sampleBtnText: {
    fontSize: 12,
    fontWeight: "700",
  },
});
