`timescale 1ns / 1ps

module gen_class (
    clk,
    nrst,
    en,

    window_hv,
    op,
    label_train,
    finish_train,

    done,
    out_hv_nonseizure,
    out_hv_seizure,
    label_predict
);
    // General parameters
    parameter DIMENSIONS = 1000;
    parameter COUNT_SIZE = 8;
    parameter HV_SEIZURE = 6'b111111;
    parameter HV_NONSEIZURE = 6'b000000;

    input clk;
    input nrst;
    input en;

    input [DIMENSIONS - 1:0] window_hv;
    input op;
    input label_train;
    input finish_train;
    
    output reg done;
    output reg [DIMENSIONS - 1:0] out_hv_nonseizure;
    output reg [DIMENSIONS - 1:0] out_hv_seizure;
    output reg label_predict;

    reg [DIMENSIONS - 1:0] hv_train;
    reg en_train;
    reg done_train;
    

    reg [DIMENSIONS - 1:0] hv_test;
    reg en_test;
    reg done_test;

    reg [1:0] state;

    cont_mem #(
        .DIMENSIONS(DIMENSIONS),
        .COUNT_SIZE(COUNT_SIZE),
        .HV_SEIZURE(HV_SEIZURE),
        .HV_NONSEIZURE(HV_NONSEIZURE)
    ) 
    u_cont_mem(
        .clk  (clk),
        .nrst  (nrst),
        .en  (en_train),
        .hv_train(hv_train),
        .label(label_train),
        .finish(finish_train),
        .done(done_train),
        .out_hv_nonseizure(out_hv_nonseizure),
        .out_hv_seizure(out_hv_seizure)
    );

    similarity #(
        .DIMENSIONS(DIMENSIONS)
    ) 
    u_similarity(
        .clk(clk),
        .nrst(nrst),
        .en(en_test),
        .hv_test(hv_test),
        .hv_nonseizure(out_hv_nonseizure),
        .hv_seizure(out_hv_seizure),
        .done(done_test),
        .label(label_predict)
    );

    always @(posedge clk or negedge nrst) begin
        if (!nrst) begin
            hv_train <= 0;
            hv_test <= 0;
            en_train <= 0;
            en_test <= 0;
            done <= 1;
            state <= 0;
        end else begin
            if (state == 0) begin
                if (en) begin
                    done <= 0;
                    if (op == 1'b0) begin
                        en_train <= 1;
                        hv_train <= window_hv;
                        state <= 1;
                    end
                    else begin
                        en_test <= 1;
                        hv_test <= window_hv;
                        state <= 2;
                    end
                end
            end 

            else if (state == 1) begin
                en_train <= 0;
                if (!en_train && done_train) begin
                    state <= 0;
                    done <= 1;
                end
                else if (en && !op) begin
                    en_train <= 1;
                    hv_train <= window_hv;
                    state <= 1;
                end
            end

            else if (state == 2) begin
                en_test <= 0;
                if (!en_test && done_test) begin
                    state <= 0;
                    done <= 1;
                end

            end
            else begin
                state <= 0;
            end
            
        end
    end
endmodule