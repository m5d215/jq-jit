def f:
    def g:
        if . == 0 then empty
        else ., (. - 1 | g) end
    ;
    [g]
;
def to_multiprecision:
    def loop:
        if . == 0 then empty
        else . % 10, ((./10) | floor | loop)
        end
    ;
    [loop]
;
