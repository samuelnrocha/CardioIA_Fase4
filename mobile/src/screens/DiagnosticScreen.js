import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { COLORS } from "../theme/colors";
import ImagePickerBox from "../components/ImagePickerBox";
import ResultCard from "../components/ResultCard";
import { analyzeChestXray } from "../services/api";

export default function DiagnosticScreen({
  mockMode,
  telemetry,
  onUpdateTelemetry,
  onBackHome,
}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [sampleType, setSampleType] = useState("cardiomegalia");
  const [sampleLabel, setSampleLabel] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handlePickCamera = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) return;
    const res = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (!res.canceled && res.assets?.[0]?.uri) {
      setSelectedImage(res.assets[0].uri);
      setSampleLabel("Exame capturado via Câmera Nativa");
    }
  };

  const handlePickGallery = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) return;
    const res = await ImagePicker.launchImageLibraryAsync({ quality: 0.9 });
    if (!res.canceled && res.assets?.[0]?.uri) {
      setSelectedImage(res.assets[0].uri);
      setSampleLabel("Exame selecionado da Galeria do Dispositivo");
    }
  };

  const handleSelectSample = (type) => {
    setSampleType(type);
    if (type === "cardiomegalia") {
      setSelectedImage("sample://cardiomegalia_sample.png");
      setSampleLabel("Amostra Clínica: Paciente ID #NIH_0042 (Cardiomegalia, ICT=0,58)");
      onUpdateTelemetry({ bpm: 134, temperatura: 38.3, queda_detectada: true });
    } else {
      setSelectedImage("sample://normal_sample.png");
      setSampleLabel("Amostra Clínica: Paciente ID #NIH_0015 (Silhueta Normal, ICT=0,43)");
      onUpdateTelemetry({ bpm: 72, temperatura: 36.5, queda_detectada: false });
    }
  };

  const handleRunCNN = async () => {
    setLoading(true);
    try {
      const response = await analyzeChestXray({
        imageUri: selectedImage || "sample://cardiomegalia_sample.png",
        sampleType,
        mockMode,
        telemetry,
      });
      setResult(response);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={onBackHome} style={styles.backBtn}>
        <Text style={styles.backText}>← Voltar ao Painel Multimodal</Text>
      </TouchableOpacity>

      <ImagePickerBox
        selectedImage={selectedImage}
        sampleLabel={sampleLabel}
        onPickCamera={handlePickCamera}
        onPickGallery={handlePickGallery}
        onSelectSample={handleSelectSample}
      />

      <TouchableOpacity
        style={styles.analyzeBtn}
        onPress={handleRunCNN}
        disabled={loading}
      >
        {loading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={COLORS.white} />
            <Text style={styles.analyzeBtnText}>
              {"  "}Executando Inferência Convolucional (ResNet18)...
            </Text>
          </View>
        ) : (
          <Text style={styles.analyzeBtnText}>
            Analisar Radiografia (CNN + Fusão Multimodal)
          </Text>
        )}
      </TouchableOpacity>

      <ResultCard result={result} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  backBtn: {
    marginBottom: 10,
  },
  backText: {
    color: COLORS.skyBlue,
    fontSize: 13,
    fontWeight: "700",
  },
  analyzeBtn: {
    backgroundColor: COLORS.fiapPink,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  analyzeBtnText: {
    color: COLORS.white,
    fontWeight: "800",
    fontSize: 14.5,
  },
});
