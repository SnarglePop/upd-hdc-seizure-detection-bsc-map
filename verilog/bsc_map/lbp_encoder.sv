`timescale 1ns / 1ps
module lbp_encoder (
    clk,
    nrst,
    en,
    
    samples,
    done,
    window_hv
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

    input clk;
    input nrst;
    input en;
    input [NUM_CHS*LBP_SIZE - 1: 0] samples;
    output reg done;
    output reg [DIMENSIONS - 1:0] window_hv;
    reg [$clog2(WINDOW_SIZE): 0] s1;
    reg signed [$clog2(WINDOW_SIZE) + 1: 0] s2;

    reg [2:0] state;


    // Windower
	// no windower


    // LBP extractor
    // no extractor

    // Item Mem
	reg [5:0] sample_pattern;
    reg [$clog2(NUM_CHS) - 1: 0] c;
    wire [DIMENSIONS - 1:0] ch_hv;
    wire [DIMENSIONS - 1:0] lbp_hv;
    item_mem_1 enc_item_mem_1(
        .c (c),
        .l (sample_pattern),
        .ch_hv (ch_hv),
        .lbp_hv  (lbp_hv)
    );

    // Binder
    wire [DIMENSIONS -1: 0] bound_ch_lbp;
    binder channel_lbp_binder (
        .hv_1(ch_hv),
        .hv_2(lbp_hv),
        .hv_out(bound_ch_lbp)
    );

                                               

    reg bundler_channel_lbp_en;
    reg [DIMENSIONS - 1: 0] bundler_channel_lbp_in;
    reg bundler_channel_lbp_finish;
    wire [DIMENSIONS - 1: 0] bundler_channel_lbp_out;

    bundler_cont  bundler_channel_lbp (
        .clk(clk),
        .nrst(nrst),
        .en(bundler_channel_lbp_en),
        .finish(bundler_channel_lbp_finish),
        .hv_in(bundler_channel_lbp_in),
        .hv_out(bundler_channel_lbp_out)
    );


    // Sample HVs bundler
    // reg [$clog2(WINDOW_SIZE) - 1: 0] s;
    reg bundler_sample_en_1;
    reg [DIMENSIONS - 1: 0] bundler_sample_in_1;
    reg bundler_sample_finish_1;
    reg bundler_sample_en_2;
    reg [DIMENSIONS - 1: 0] bundler_sample_in_2;
    reg bundler_sample_finish_2;
    reg delay;
    reg first_sample;
    wire [DIMENSIONS - 1: 0] bundler_sample_out_1;
    wire [DIMENSIONS - 1: 0] bundler_sample_out_2;
    reg complete_window;

    bundler_cont bundler_sample_1 (
        .clk(clk),
        .nrst(nrst),
        .en(bundler_sample_en_1),
        .finish(bundler_sample_finish_1),
        .hv_in(bundler_sample_in_1),
        .hv_out(bundler_sample_out_1)
    );
    
    bundler_cont bundler_sample_2 (
        .clk(clk),
        .nrst(nrst),
        .en(bundler_sample_en_2),
        .finish(bundler_sample_finish_2),
        .hv_in(bundler_sample_in_2),
        .hv_out(bundler_sample_out_2)
    );


    always_ff @(posedge clk or negedge nrst) begin
        if (!nrst) begin
            done <= 0;
            window_hv <= 0;
            state <= 0;
            sample_pattern <= 0;

            c <= 0;

            bundler_channel_lbp_en <= 0;
            bundler_channel_lbp_finish <= 0;
            bundler_channel_lbp_in <= 0;

            s1 <= 0;
            s2 <= 0;
            bundler_sample_en_1 <= 0;
            bundler_sample_finish_1 <= 0;
            bundler_sample_in_1 <= 0;
            bundler_sample_en_2 <= 0;
            bundler_sample_finish_2 <= 0;
            bundler_sample_in_2 <= 0;
            delay <= 0;
            first_sample <= 1;
            complete_window <= 0;
            // first_count <= 0;

        end else begin
            if (state == 0) begin
                done <= 0;
                bundler_sample_en_1 <= 0;
                bundler_sample_en_2 <= 0;
                if (en) begin
                    state <= 1;
                    c <= 0;
                end
            end

            // Compute LBP pattern
            else if (state == 1) begin
                sample_pattern <= samples[c*LBP_SIZE +: LBP_SIZE];
                bundler_channel_lbp_en <= 0;
                state <= 2;
            end

            // Iterate through each channel, bind and add to channel-LBP bundler
            else if (state == 2) begin
                bundler_channel_lbp_in <= bound_ch_lbp;
                bundler_channel_lbp_en <= 1;
                bundler_channel_lbp_finish <= 0;
                
                if (c == NUM_CHS - 1) begin
                    c <= 0;
                    state <= 3;
                end 
                else begin
                    c <= c + 1;
                    state <= 1;
                end
            end

            else if (state == 3) begin
                bundler_channel_lbp_en <= 0;
                bundler_channel_lbp_finish <= 1;

                state <= 4;
            end

            // Finish channel-LBP bundler and iterate through samples
            else if (state == 4) begin
                bundler_channel_lbp_finish <= 0;
                bundler_sample_en_1 <= 1;
                bundler_sample_in_1 <= bundler_channel_lbp_out;
                if (!first_sample) begin
                    bundler_sample_en_2 <= 1;
                    bundler_sample_in_2 <= bundler_channel_lbp_out;
                end
                if (s1 == WINDOW_SIZE) begin
                    s1 <= 1;
                    complete_window <= 0;
                    s2 <= s2 + 1;
                    state <= 5;
                end 
                else if (s2 == WINDOW_SIZE) begin
                    s2 <= 1;
                    complete_window <= 1;
                    s1 <= s1 + 1;
                    state <= 5;
                end 
                else if (first_sample) begin
                    if (s2 == WINDOW_STEP) begin
                        first_sample <= 0;
                        s2 <= 1;
                        s1 <= s1 + 1;
                    end
                    else begin
                        s2 <= s2 + 1;
                    end
                    s1 <= s1 + 1;
                    state <= 0;
                end 
                else begin
                    s1 <= s1 + 1;
                    s2 <= s2 + 1;
                    state <= 0;
                end
            end

            // Finish sample bundler
            else if (state == 5) begin
                bundler_sample_en_1 <= 0;
                bundler_sample_en_2 <= 0;
                if (complete_window) begin
                    bundler_sample_finish_2 <= 1;
                end else begin
                    bundler_sample_finish_1 <= 1;
                end
                state <= 6;
            end

            // Output window HV
            else if (state == 6) begin
                if (complete_window) begin
                    bundler_sample_finish_2 <= 0;
                end else begin
                    bundler_sample_finish_1 <= 0;
                end
                state <= 7;
            end
            else if (state == 7) begin
                if (complete_window) begin
                    window_hv <= bundler_sample_out_2;
                end else begin
                    window_hv <= bundler_sample_out_1;
                end
                done <= 1;
                state <= 0;
            end
        end
    end

endmodule
