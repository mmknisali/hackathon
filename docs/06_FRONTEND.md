# Frontend Documentation

This document explains how the frontend works and how it integrates with the API.

## Overview

The frontend is a single HTML file (`frontend/index.html`) that provides:
- Interactive demo panel for error correction
- Real-time visualization with Plotly
- Model statistics display
- API connection status

## File Structure

```
frontend/
└── index.html    # Single-page demo application
```

No build step required - just open in a browser!

## Opening the Frontend

```bash
# Using Chrome
google-chrome frontend/index.html

# Using Firefox
firefox frontend/index.html

# Using VS Code
code frontend/index.html
```

Or simply double-click the file in your file manager.

## How It Works

### 1. UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    HEADER                                    │
│  Star Tracker Sensor Error Correction                        │
│  LSTM Autoencoder Neural Network                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │   LIVE DEMO PANEL   │  │     MODEL STATISTICS        │  │
│  │                     │  │  MSE Before │ MSE After     │  │
│  │  [Input Data]       │  │  MAE Before │ MAE After     │  │
│  │  [Generate Sample]  │  │  Error Reduction %          │  │
│  │  [Correct Errors]   │  │                              │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              BEFORE / AFTER VISUALIZATION                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Plotly Chart (Before)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Plotly Chart (After)                    │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                   CONNECTION STATUS                           │
└─────────────────────────────────────────────────────────────┘
```

### 2. Components

#### Header
- Title: "Star Tracker Sensor Error Correction"
- Subtitle: Technical description
- NASA-style badge

#### Live Demo Panel
- Textarea for input data (JSON format)
- "Generate Sample" button - creates random test data
- "Correct Errors" button - sends to API and gets results

#### Model Statistics
- MSE Before correction
- MSE After correction
- MAE Before correction
- MAE After correction
- Error Reduction percentage

#### Before/After Charts
- Two Plotly line charts
- Shows X coordinate over timesteps
- Red: Corrupted input
- Green: Corrected output

#### Connection Status
- Shows if API is connected
- Green indicator = connected
- Red indicator = not connected

## How It Calls the API

### 1. Check Connection

```javascript
async function checkConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            // Show connected status
        }
    } catch (e) {
        // Show disconnected status
    }
}
```

### 2. Generate Sample Data

```javascript
function generateSampleData() {
    const sequence = [];
    for (let i = 0; i < 50; i++) {
        // Generate random star tracker data with noise
        const baseX = Math.sin(i * 0.1) * 0.8;
        const noise = (Math.random() - 0.5) * 0.3;
        const cosmicSpike = Math.random() < 0.05 ? (Math.random() - 0.5) * 2 : 0;
        
        sequence.push([
            baseX + noise + cosmicSpike,  // x
            Math.cos(i * 0.1) * 0.5,       // y
            0.3,                            // z
            0.7 + Math.random() * 0.1,     // q1
            0.2 + Math.random() * 0.1,     // q2
            0.3 + Math.random() * 0.1,     // q3
            0.5 + Math.random() * 0.1      // q4
        ]);
    }
    return sequence;
}
```

### 3. Send to API

```javascript
async function correctErrors() {
    // Get input from textarea
    const inputData = JSON.parse(document.getElementById('inputData').value);
    
    // Call API
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: inputData })
    });
    
    const result = await response.json();
    const corrected = result.corrected;
    
    // Update charts
    plotData('beforeChart', 'Corrupted Input', inputData.slice(0, 30), '#ff4444');
    plotData('afterChart', 'Corrected Output', corrected.slice(0, 30), '#00ff88');
}
```

### 4. Plot Data with Plotly

```javascript
function plotData(containerId, title, data, color) {
    const xData = data.map((d, i) => i);
    const yData = data.map(d => d[0]);  // X coordinate
    
    const trace = {
        x: xData,
        y: yData,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: color, width: 2 }
    };
    
    const layout = {
        title: { text: title, font: { color: '#00d4ff' } },
        paper_bgcolor: '#0a0a1a',
        plot_bgcolor: '#0a0a1a',
        font: { color: '#888' }
    };
    
    Plotly.newPlot(containerId, [trace], layout);
}
```

## Demo Mode

If the API is not running, the frontend runs in "Demo Mode":
- Uses a simple smoothing algorithm instead of the model
- Shows simulated correction results
- Allows testing the UI without the backend

## Customization

### Change API URL

```javascript
const API_URL = 'http://localhost:8000';  // Change this
```

### Change Chart Colors

```javascript
// In plotData function
line: { color: '#ff4444', width: 2 }  // Red for before
line: { color: '#00ff88', width: 2 }  // Green for after
```

### Change Sample Data Generation

```javascript
// In generateSampleData function
// Modify noise levels, sequence length, etc.
```

## Troubleshooting

### API Not Connecting

1. Make sure API is running: `cd backend && python api.py`
2. Check the URL in JavaScript matches: `http://localhost:8000`
3. Check firewall settings

### Charts Not Updating

1. Check browser console for errors
2. Verify JSON format is correct
3. Make sure Plotly is loading (check CDN)

## Key Files

- `frontend/index.html` - Complete frontend application
- `docs/05_API.md` - API endpoints used
