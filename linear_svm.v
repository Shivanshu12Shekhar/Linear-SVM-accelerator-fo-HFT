`timescale 1ns / 1ps

module linear_svm #(
    parameter FEATURE_DIM = 16,
    parameter DATA_WIDTH = 16,
    parameter WEIGHT_WIDTH = 16,
    parameter BIAS_WIDTH = 32,
    parameter FRAC_BITS = 11, // Q5.11
    parameter ACCUM_WIDTH = 36 // DATA(16) + WEIGHT(16) + LOG2(16)(4) = 36
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [FEATURE_DIM*DATA_WIDTH-1:0] data_in, // Flattened input: feat[15]...feat[0]
    output reg class_out, // 1 for +1 (BUY), 0 for -1 (SELL)
    output reg valid_out
);

    // --- Internal Signals ---
    
    // Weights and Bias Memory
    reg signed [WEIGHT_WIDTH-1:0] weights [0:FEATURE_DIM-1];
    reg signed [BIAS_WIDTH-1:0] bias_rom [0:0];
    
    // ----------------------------------------------------
    // Initialization (Load Weights/Bias)
    // Using $readmemh for synthesis (Vivado supports this for ROM inference)
    // ----------------------------------------------------
    initial begin
        $readmemh("weights.mem", weights);
        $readmemh("bias.mem", bias_rom);
    end

    // Bias assigned once at initialization (ROM-style) ensures it remains stable
    wire signed [BIAS_WIDTH-1:0] bias = bias_rom[0];

    // Input Registers
    reg signed [DATA_WIDTH-1:0] data_reg [0:FEATURE_DIM-1];
    reg valid_in_reg;

    // Multiplier Outputs (Extended Width for Accumulation)
    reg signed [ACCUM_WIDTH-1:0] mult_out [0:FEATURE_DIM-1];
    
    // ----------------------------------------------------
    // Stage 1: Input Registration & Unpacking
    // ----------------------------------------------------
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_in_reg <= 0;
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                data_reg[i] <= 0;
            end
        end else begin
            valid_in_reg <= valid_in;
            if (valid_in) begin
                for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                    // Unpack flattened input
                    data_reg[i] <= $signed(data_in[i*DATA_WIDTH +: DATA_WIDTH]); 
                end
            end
        end
    end

    // ----------------------------------------------------
    // Stage 2: Parallel Multiplication & Scaling
    // ----------------------------------------------------
    reg valid_mult;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_mult <= 0;
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                mult_out[i] <= 0;
            end
        end else begin
            valid_mult <= valid_in_reg;
            for (i = 0; i < FEATURE_DIM; i = i + 1) begin
                mult_out[i] <= (data_reg[i] * weights[i]);
            end
        end
    end

    // ----------------------------------------------------
    // Stage 3: Adder Tree (2-cycle latency)
    // ----------------------------------------------------
    // Since building a full 16-input adder tree in one cycle might fail timing at high Fmax,
    // and deep pipelining (4 cycles) is overkill, we pipeline it into 2 cycles.
    // Cycle 1: Levels 1 & 2 (16 -> 8 -> 4)
    // Cycle 2: Levels 3 & 4 (4 -> 2 -> 1)
    
    // Combinational intermediate signals
    wire signed [ACCUM_WIDTH-1:0] comb_stg1 [0:7];
    wire signed [ACCUM_WIDTH-1:0] comb_stg2 [0:3];
    wire signed [ACCUM_WIDTH-1:0] comb_stg3 [0:1];
    wire signed [ACCUM_WIDTH-1:0] comb_final;

    // Registers for pipeline
    reg signed [ACCUM_WIDTH-1:0] reg_stg2 [0:3];
    reg signed [ACCUM_WIDTH-1:0] adder_tree_out;
    
    // Pipeline valid signals
    reg valid_tree_stg1, valid_tree_out;

    // --- Combinational Level 1 (16 -> 8) ---
    assign comb_stg1[0] = mult_out[0] + mult_out[1];
    assign comb_stg1[1] = mult_out[2] + mult_out[3];
    assign comb_stg1[2] = mult_out[4] + mult_out[5];
    assign comb_stg1[3] = mult_out[6] + mult_out[7];
    assign comb_stg1[4] = mult_out[8] + mult_out[9];
    assign comb_stg1[5] = mult_out[10] + mult_out[11];
    assign comb_stg1[6] = mult_out[12] + mult_out[13];
    assign comb_stg1[7] = mult_out[14] + mult_out[15];

    // --- Combinational Level 2 (8 -> 4) ---
    assign comb_stg2[0] = comb_stg1[0] + comb_stg1[1];
    assign comb_stg2[1] = comb_stg1[2] + comb_stg1[3];
    assign comb_stg2[2] = comb_stg1[4] + comb_stg1[5];
    assign comb_stg2[3] = comb_stg1[6] + comb_stg1[7];

    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_tree_stg1 <= 0;
            for (j = 0; j < 4; j = j + 1) reg_stg2[j] <= 0;
        end else begin
            valid_tree_stg1 <= valid_mult;
            // Register at the mid-point of the tree
            for (j = 0; j < 4; j = j + 1) reg_stg2[j] <= comb_stg2[j];
        end
    end

    // --- Combinational Level 3 (4 -> 2) ---
    assign comb_stg3[0] = reg_stg2[0] + reg_stg2[1];
    assign comb_stg3[1] = reg_stg2[2] + reg_stg2[3];

    // --- Combinational Level 4 (2 -> 1) ---
    assign comb_final = comb_stg3[0] + comb_stg3[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_tree_out <= 0;
            adder_tree_out <= 0;
        end else begin
            valid_tree_out <= valid_tree_stg1;
            adder_tree_out <= comb_final;
        end
    end

    // ----------------------------------------------------
    // Stage 4: Bias Add & Output Logic
    // ----------------------------------------------------
    // Add bias combinationally (sign extension happens naturally)
    wire signed [ACCUM_WIDTH-1:0] dot_product_with_bias = adder_tree_out + bias;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            class_out <= 0;
        end else begin
            valid_out <= valid_tree_out;
            
            // Decision Function: sign(dot_product + bias)
            // If >= 0, Class +1 (1), else Class -1 (0)
            if (dot_product_with_bias >= 0)
                class_out <= 1;
            else
                class_out <= 0;
        end
    end

endmodule
