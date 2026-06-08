`timescale 1ns / 1ps
module tb_lbp_encoder_max ();
    localparam T = 1000;  // clock period in ns

    // General params
    localparam DIMENSIONS = 10000;
    localparam NUM_CHS = 2;
    localparam WINDOW_SIZE = 4;
    localparam WINDOW_STEP = 2;
    localparam SAMPLE_SIZE = 6;

    // LBP params
    localparam LBP_SIZE = 6;
    localparam NUM_LBP = 64;

    // Bundler params
    localparam COUNT_SIZE = 8;

    reg clk;
    reg nrst;
    reg en;
    reg [NUM_CHS - 1: 0][SAMPLE_SIZE - 1: 0] samples;
    wire done;
    wire [DIMENSIONS - 1:0] window_hv;

    lbp_encoder #(
        .DIMENSIONS(DIMENSIONS),
        .NUM_CHS(NUM_CHS),
        .WINDOW_SIZE(WINDOW_SIZE),
        .WINDOW_STEP(WINDOW_STEP),
        .SAMPLE_SIZE(SAMPLE_SIZE),
        .LBP_SIZE(LBP_SIZE),
        .NUM_LBP(NUM_LBP)
    )
    u_lbp_encoder(   
        .clk (clk),
        .nrst (nrst),
        .en (en),
        .samples (samples),
        .done (done),
        .window_hv (window_hv)
    );

    // Clock
    always begin
        clk = 1'b1;
        #(T / 2);
        clk = 1'b0;
        #(T / 2);
    end

    initial begin
        // $vcdplusfile("tb_lbp_encoder.vpd");
        // $vcdpluson;
        nrst = 0;
        en = 0;
        samples = 0;
        # (10000 - 500)
        nrst = 1;

        en = 1;
		samples[0] = 6'b001010;
		samples[1] = 6'b101111;
		# (5 + 5)
		en = 0;
		# (200 - 5 - 5)

        wait (u_lbp_encoder.done == 1);
        # 2000
        $finish;
    end
endmodule