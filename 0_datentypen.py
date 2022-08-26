
def datentyp_str_int():
    a = '1'
    
    b = '2'
    
    c = a + b
    
    c_int = int(a) + int(b)
    
    print(f'a + b = {c}')
    
    print(f'int(a) + int(b) = {c_int}')
    
    try:
        c + 5
    except TypeError as exc:
        print(f'Exception: {exc}')
        
        c + str(5)
        
    
    c_int + 5
    return c_int
    
if __name__ == '__main__':
    c_int = datentyp_str_int()
    print(f'Datentyp c_int: {type(c_int)}, wert: {c_int}')


