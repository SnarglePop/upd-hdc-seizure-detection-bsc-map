`timescale 1ns / 100ps
module lbp_extractor (
    sample_window,
    sample_pattern
);

    // General params
    parameter SAMPLE_SIZE = 16;
    parameter LBP_SIZE = 6;

    input [SAMPLE_SIZE*(LBP_SIZE+1)-1: 0] sample_window;
    output logic [LBP_SIZE - 1: 0] sample_pattern;
    
    //reg signed [SAMPLE_SIZE - 1:0] left;
    //reg signed [SAMPLE_SIZE - 1:0] right;

    always_comb begin
        for (int i = 0; i < LBP_SIZE; i = i + 1) begin
            if ($signed(sample_window[i*SAMPLE_SIZE +: SAMPLE_SIZE]) <= $signed(sample_window[(i+1)*SAMPLE_SIZE +: SAMPLE_SIZE]))
                sample_pattern[LBP_SIZE - 1 - i] = 1'b1;
            else
                sample_pattern[LBP_SIZE - 1 - i] = 1'b0;
        end
    end

endmodule