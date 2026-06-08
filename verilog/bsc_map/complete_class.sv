`timescale 1ns / 1ps
module complete_class (
    clk,
    nrst,
    en,
    
    samples,

    op,
    label_train,
    finish_train,
    
    done,
    label_predict
);

    // General params
    parameter DIMENSIONS = 2000;
    parameter NUM_CHS = 17;
    parameter WINDOW_SIZE = 256;
    parameter WINDOW_STEP = 128;
    parameter SAMPLE_SIZE = 16;
    
    // Feature params
    parameter LBP_SIZE = 6;
    parameter NUM_LBP = 64;

    // Bundler params
    parameter COUNT_SIZE = 8;
    
    // genclass params
    parameter HV_SEIZURE = 6'b111111;
    parameter HV_NONSEIZURE = 6'b000000;
    
    input clk;
    input nrst;
    input en;
    input [LBP_SIZE*NUM_CHS - 1: 0] samples;
    
    input op;
    input label_train;
    input finish_train;
    
    output reg done;
    reg [DIMENSIONS - 1:0] out_hv_nonseizure;
    reg [DIMENSIONS - 1:0] out_hv_seizure;
    output reg label_predict;
    reg [$clog2(WINDOW_SIZE): 0] s;
    reg [(WINDOW_SIZE + LBP_SIZE)*(NUM_CHS)*(SAMPLE_SIZE) - 1: 0] sample_memory;
    
    wire done_mid;
    wire [DIMENSIONS - 1:0] window_hv_mid;
    
    // LBP Encoder
    lbp_encoder #(
        .DIMENSIONS(DIMENSIONS),
        .NUM_CHS(NUM_CHS),
        .WINDOW_SIZE(WINDOW_SIZE),
        .WINDOW_STEP(WINDOW_STEP),
        .SAMPLE_SIZE(SAMPLE_SIZE),
        .LBP_SIZE(LBP_SIZE),
        .NUM_LBP(NUM_LBP)
    )
    u_lbp_encoder (
        .clk(clk),
        .nrst(nrst),
        .en(en),
        .samples(samples),
        .done(done_mid),
        .window_hv(window_hv_mid)
    );
    // Gen Class
    gen_class #(
        .DIMENSIONS(DIMENSIONS),
        .COUNT_SIZE(COUNT_SIZE),
        .HV_SEIZURE(HV_SEIZURE),
        .HV_NONSEIZURE(HV_NONSEIZURE)
    ) 
    u_gen_class (
        .clk(clk),
        .nrst(nrst),
        .en(done_mid),
        .window_hv(window_hv_mid),
        .op(op),
        .label_train(label_train),
        .finish_train(finish_train),
        .done(done),
        .out_hv_nonseizure(out_hv_nonseizure),
        .out_hv_seizure(out_hv_seizure),
        .label_predict(label_predict)
    );
    
    
    
endmodule
