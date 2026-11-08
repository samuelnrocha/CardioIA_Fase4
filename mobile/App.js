import React, { useState } from "react";
import { SafeAreaView, ScrollView, StatusBar, StyleSheet } from "react-native";
import { COLORS } from "./src/theme/colors";
import Header from "./src/components/Header";
import HomeScreen from "./src/screens/HomeScreen";
import DiagnosticScreen from "./src/screens/DiagnosticScreen";

export default function App() {
  const [currentScreen, setCurrentScreen] = useState("diagnostic");
  const [mockMode, setMockMode] = useState(false);
  const [telemetry, setTelemetry] = useState({
    bpm: 76,
    temperatura: 36.6,
    queda_detectada: false,
  });

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.background} />
      <Header mockMode={mockMode} />
      <ScrollView contentContainerStyle={styles.scroll}>
        {currentScreen === "home" ? (
          <HomeScreen
            mockMode={mockMode}
            onToggleMock={setMockMode}
            telemetry={telemetry}
            onNavigateDiagnostic={() => setCurrentScreen("diagnostic")}
          />
        ) : (
          <DiagnosticScreen
            mockMode={mockMode}
            telemetry={telemetry}
            onUpdateTelemetry={setTelemetry}
            onBackHome={() => setCurrentScreen("home")}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scroll: {
    paddingBottom: 32,
  },
});
