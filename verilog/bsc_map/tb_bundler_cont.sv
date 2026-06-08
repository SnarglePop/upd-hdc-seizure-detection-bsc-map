`timescale 1ns / 1ps
module tb_bundler_cont ();
    localparam T = 10;  // clock period in ns


    parameter DIMENSIONS = 6;
    parameter COUNT_SIZE = 3;

    reg clk;
    reg nrst;
    reg en;
    reg [DIMENSIONS - 1:0] hv_in;
    reg finish;
    wire [DIMENSIONS - 1:0] hv_out;

    bundler_cont #(
        .DIMENSIONS(DIMENSIONS),
        .COUNT_SIZE(COUNT_SIZE)
    ) 
    u_bundler_cont (
        .clk  (clk),
        .nrst  (nrst),
        .en (en),
        .hv_in (hv_in),
        .finish (finish),
        .hv_out (hv_out)
    );

    // Clock
    always begin
        clk = 1'b1;
        #(T / 2);
        clk = 1'b0;
        #(T / 2);
    end

    initial begin
        nrst = 0;
        en = 0;
        finish = 0;
        hv_in = 6'b0;
        # (20 - 5)
        nrst = 1;

        en = 1;
        hv_in = 6'b100101;
        # (5 + 5)
        en = 0;
        # (60 - 5 - 5)

        en = 1;
        hv_in = 6'b110001;
        # (5 + 5)
        en = 0;
        # (60 - 5 - 5)

        en = 1;
        hv_in = 6'b111111;
        # (5 + 5)
        en = 0;
        finish = 1;
        # (5+5)
        finish = 0;
        # (60 - 5)
        $finish;
    end
endmodule