#  Hardware-Accelerated Linear SVM for Low-Latency High-Frequency Trading Signal Classification

This project implements a **fully pipelined Linear SVM accelerator on FPGA (PYNQ-Z2)** for **ultra-low latency financial signal classification**, targeting high-frequency trading (HFT) applications.

The design combines **machine learning, fixed-point optimization, and FPGA-based hardware acceleration** to achieve real-time inference with high throughput.

---

##  Key Features

- Fully pipelined **Linear SVM accelerator in Verilog**
- **16 parallel multipliers + pipelined reduction tree**
- Fixed-point implementation (**Q5.11**) with minimal accuracy loss (<0.1%)
- High-speed data transfer using **AXI-Stream + AXI DMA**
- End-to-end pipeline: **Python (training) → FPGA (inference)**
- Validated with **100+ test vectors**
- Operates at ~**150 MHz**
---

## Training & Data Flow

1. Load dataset (financial features)
2. Preprocess:
   - Missing value imputation
   - Standardization
   - PCA → 16 features
3. Train Linear SVM (scikit-learn)
4. Export:
   - weights.mem
   - bias.mem
5. Generate:
   - input_vectors.mem
   - expected_outputs.mem

---

## Hardware Design

### Pipeline Stages

1. Input register & unpacking  
2. Parallel multiplication (16 features)  
3. 2-stage pipelined adder tree  
4. Bias addition + decision logic  

---

## ✅ Verification

- Compared FPGA output with software model  
- Tested on **100+ input vectors**  
- Achieved **cycle-accurate matching**  

---
