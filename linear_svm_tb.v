`timescale 1ns / 1ps
module linear_svm_tb;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    parameter FEATURE_DIM  = 16;
    parameter DATA_WIDTH   = 16;
    parameter WEIGHT_WIDTH = 16;
    parameter BIAS_WIDTH   = 32;
    parameter CLK_PERIOD   = 10;          // 100 MHz
    parameter MAX_WAIT_CYC = 1000;        // Watchdog: max cycles waiting for valid_out

    // -------------------------------------------------------------------------
    // DUT Signals
    // -------------------------------------------------------------------------
    reg  clk;
    reg  rst_n;
    reg  valid_in;
    reg  [FEATURE_DIM*DATA_WIDTH-1:0] data_in;
    wire class_out;
    wire valid_out;

    // -------------------------------------------------------------------------
    // Test Variables
    // -------------------------------------------------------------------------
    integer f_in, f_out_exp, f_true;       // file handles
    integer scan_in, scan_exp, scan_true;

    reg [FEATURE_DIM*DATA_WIDTH-1:0] test_vector;
    reg expected_class;                    // HW-predicted label from Python sim
    reg true_class;                        // ground-truth label

    integer mismatch_count;
    integer test_count;
    integer wait_cyc;
    integer timeout_count;

    // -------------------------------------------------------------------------
    // DUT Instantiation
    // -------------------------------------------------------------------------
    linear_svm #(
        .FEATURE_DIM (FEATURE_DIM),
        .DATA_WIDTH  (DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .BIAS_WIDTH  (BIAS_WIDTH)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in),
        .data_in  (data_in),
        .class_out(class_out),
        .valid_out(valid_out)
    );

    // -------------------------------------------------------------------------
    // Clock Generation
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Task: wait for valid_out with watchdog timeout
    // -------------------------------------------------------------------------
    task wait_for_valid_out;
        output timed_out;
        integer cyc;
        begin
            timed_out = 0;
            cyc       = 0;
            while (!valid_out && cyc < MAX_WAIT_CYC) begin
                @(posedge clk);
                cyc = cyc + 1;
            end
            if (cyc >= MAX_WAIT_CYC) begin
                $display("[WATCHDOG] Test %0d: valid_out never asserted after %0d cycles!",
                         test_count, MAX_WAIT_CYC);
                timed_out = 1;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Main Test Procedure
    // -------------------------------------------------------------------------
    initial begin
        // Init
        rst_n         = 0;
        valid_in      = 0;
        data_in       = 0;
        mismatch_count = 0;
        test_count     = 0;
        timeout_count  = 0;

        // Open test vector files
        f_in      = $fopen("input_vectors.mem",    "r");
        f_out_exp = $fopen("expected_outputs.mem", "r");
        f_true    = $fopen("true_labels.mem",      "r");

        if (f_in == 0 || f_out_exp == 0 || f_true == 0) begin
            $display("ERROR: Could not open one or more test vector files.");
            $display("  input_vectors.mem    : %0d", f_in);
            $display("  expected_outputs.mem : %0d", f_out_exp);
            $display("  true_labels.mem      : %0d", f_true);
            $finish;
        end

        // Reset sequence
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("=================================================================");
        $display(" Linear SVM Testbench — 100 Test Vectors");
        $display("=================================================================");
        $display(" Label encoding : 1 = UP (+1),  0 = DOWN (-1)");
        $display(" Columns        : Test# | True | Expected(PySim) | DUT_Out | Status");
        $display("-----------------------------------------------------------------");

        // -----------------------------------------------------------------
        // Main loop — one vector per iteration
        // -----------------------------------------------------------------
        scan_in   = $fscanf(f_in,      "%h\n", test_vector);
        scan_exp  = $fscanf(f_out_exp, "%d\n", expected_class);
        scan_true = $fscanf(f_true,    "%d\n", true_class);

        while (scan_in == 1 && scan_exp == 1 && scan_true == 1) begin

            // ---- Drive input for exactly one clock cycle ------------------
            @(negedge clk);          // drive on falling edge ? stable at next posedge
            valid_in <= 1;
            data_in  <= test_vector;

            @(negedge clk);          // de-assert after one cycle
            valid_in <= 0;

            // ---- Wait for DUT to assert valid_out (with watchdog) ---------
            begin : wait_block
                reg timed_out;
                wait_for_valid_out(timed_out);

                if (timed_out) begin
                    timeout_count = timeout_count + 1;
                    $display("  [%3d]  true=%0d  expected=%0d  dut=X  TIMEOUT",
                             test_count, true_class, expected_class);
                end else begin
                    // Capture result synchronously (valid_out is high now)
                    // Compare DUT output against expected (Python fixed-pt sim)
                    if (class_out !== expected_class) begin
                        mismatch_count = mismatch_count + 1;
                        $display("  [%3d]  true=%0d  expected=%0d  dut=%0d  MISMATCH",
                                 test_count, true_class, expected_class, class_out);
                    end else begin
                        $display("  [%3d]  true=%0d  expected=%0d  dut=%0d  OK",
                                 test_count, true_class, expected_class, class_out);
                    end
                end
            end

            test_count = test_count + 1;

            // ---- Wait for valid_out to deassert before next vector ---------
            @(posedge clk);
            if (valid_out) @(posedge clk);   // extra cycle if still high

            // ---- Read next entry ------------------------------------------
            scan_in   = $fscanf(f_in,      "%h\n", test_vector);
            scan_exp  = $fscanf(f_out_exp, "%d\n", expected_class);
            scan_true = $fscanf(f_true,    "%d\n", true_class);
        end

        // -----------------------------------------------------------------
        // Final report
        // -----------------------------------------------------------------
        $display("=================================================================");
        $display(" Simulation Complete");
        $display("  Total vectors tested : %0d", test_count);
        $display("  DUT vs PySim mismatches : %0d", mismatch_count);
        $display("  Timeouts             : %0d", timeout_count);
        $display("-----------------------------------------------------------------");
        if (mismatch_count == 0 && timeout_count == 0)
            $display(" STATUS: PASS — DUT matches Python fixed-point simulation exactly");
        else
            $display(" STATUS: FAIL — see mismatches above");
        $display("=================================================================");

        $fclose(f_in);
        $fclose(f_out_exp);
        $fclose(f_true);
        $finish;
    end

endmodule