import type { Config } from 'tailwindcss';

// Sistema visual "Mission Control": consola oscura de telemetría en tiempo real.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0A0E14', // fondo de la app (casi negro azulado)
        surface: '#111823', // paneles
        'surface-2': '#0E141D', // paneles hundidos / sidebar
        line: '#1E2A3A', // bordes
        ink: '#E6EDF3', // texto primario
        muted: '#7D8DA1', // texto secundario
        dim: '#42546B', // texto terciario / ejes
        primary: '#22D3EE', // acento cian (telemetría)
        normal: '#34D399', // señal normal
        warning: '#FBBF24', // warmup / advertencia
        anomaly: '#FB5E5E', // anomalía
      },
      fontFamily: {
        display: ['"Chakra Petch"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        sans: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'glow-normal': '0 0 24px -6px rgba(52,211,153,0.55)',
        'glow-anomaly': '0 0 28px -4px rgba(251,94,94,0.6)',
        'glow-warning': '0 0 24px -6px rgba(251,191,36,0.55)',
        'glow-primary': '0 0 24px -6px rgba(34,211,238,0.5)',
        panel: 'inset 0 1px 0 0 rgba(255,255,255,0.03)',
      },
      keyframes: {
        pulseDot: {
          '0%,100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.35', transform: 'scale(0.82)' },
        },
        glowPulse: {
          '0%,100%': { opacity: '1' },
          '50%': { opacity: '0.72' },
        },
        sweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        pulseDot: 'pulseDot 1.6s ease-in-out infinite',
        glowPulse: 'glowPulse 1.8s ease-in-out infinite',
        sweep: 'sweep 2.4s linear infinite',
      },
    },
  },
  plugins: [],
} satisfies Config;
