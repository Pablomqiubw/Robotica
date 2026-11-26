MODULE ROS_TcpMover
    ! Servidor TCP seguro en Manual Reducido (puerto 5005 en Service IP 192.168.125.1).
    ! Comandos <= 80 chars:
    !   "PING"               -> "PONG"
    !   "GET"                -> "[j1,j2,j3,j4,j5,j6]"
    !   "V:[v_tcp,v_ori,v_l,v_r]" -> ajusta velocidad segura (v_tcp 5..150 mm/s, v_ori 5..60 deg/s)
    !   "[j1,j2,j3,j4,j5,j6]"     -> mueve MoveAbsJ validando limites, paso <= 20 deg y Z >= 0
    LOCAL CONST string tcpHost := "192.168.125.1";
    LOCAL CONST num tcpPort := 5005;
    LOCAL CONST num margen := 2;
    LOCAL CONST num maxPasoDeg := 20;
    LOCAL CONST num alturaMinima := 0;
    LOCAL VAR speeddata vTcpActual := [30,15,30,15];

    LOCAL VAR socketdev srvSock;
    LOCAL VAR socketdev cliSock;

    PROC ServidorTCP()
        VAR num resp;
        VAR errnum motivo;
        VAR string clientIp;

        IF OpMode() <> OP_MAN_PROG THEN
            TPWrite "Seleccione Manual Reducido (< 250 mm/s).";
            RETURN;
        ENDIF

        TPReadFK resp,"Aceptar objetivos TCP de Ubuntu?",stEmpty,stEmpty,stEmpty,"Iniciar","Cancelar"\MaxTime:=30\BreakFlag:=motivo;
        IF motivo <> 0 OR resp <> 4 THEN
            TPWrite "Servidor TCP cancelado.";
            RETURN;
        ENDIF

        IF SocketGetStatus(cliSock) <> SOCKET_CLOSED THEN
            SocketClose cliSock;
        ENDIF
        IF SocketGetStatus(srvSock) <> SOCKET_CLOSED THEN
            SocketClose srvSock;
        ENDIF

        SocketCreate srvSock;
        SocketBind srvSock, tcpHost, tcpPort;
        SocketListen srvSock;
        TPWrite "ServidorTCP escuchando en puerto "\Num:=tcpPort;

        WHILE TRUE DO
            IF SocketGetStatus(cliSock) <> SOCKET_CLOSED THEN
                SocketClose cliSock;
            ENDIF
            SocketAccept srvSock, cliSock, \ClientAddress:=clientIp;
            TPWrite "PC conectada: " + clientIp;
            AtenderCliente;
        ENDWHILE

    UNDO
        IF SocketGetStatus(cliSock) <> SOCKET_CLOSED THEN
            SocketClose cliSock;
        ENDIF
        IF SocketGetStatus(srvSock) <> SOCKET_CLOSED THEN
            SocketClose srvSock;
        ENDIF
    ENDPROC

    LOCAL PROC AtenderCliente()
        VAR string msg;
        VAR jointtarget objetivo;
        VAR jointtarget actual;
        VAR robtarget poseDestino;
        VAR num codigo;
        VAR speeddata spd;
        VAR bool okSpd;
        VAR string prefijo;

        WHILE TRUE DO
            msg := "";
            SocketReceive cliSock, \Str:=msg;
            IF msg = "PING" THEN
                SocketSend cliSock, \Str:="PONG";
            ELSEIF msg = "GET" THEN
                actual := CJointT();
                SocketSend cliSock, \Str:=ValToStr(actual.robax);
            ELSE
                prefijo := "";
                IF StrLen(msg) >= 4 THEN
                    prefijo := StrPart(msg, 1, 2);
                ENDIF
                IF prefijo = "V:" THEN
                    okSpd := StrToVal(StrPart(msg, 3, StrLen(msg) - 2), spd);
                    IF okSpd = TRUE THEN
                        IF spd.v_tcp >= 5 AND spd.v_tcp <= 150 AND spd.v_ori >= 5 AND spd.v_ori <= 60 THEN
                            vTcpActual := [spd.v_tcp, spd.v_ori, spd.v_tcp, spd.v_ori];
                            TPWrite "Velocidad TCP mm/s = "\Num:=vTcpActual.v_tcp;
                            SocketSend cliSock, \Str:="OK";
                        ELSE
                            SocketSend cliSock, \Str:="ERR:-6";
                        ENDIF
                    ELSE
                        SocketSend cliSock, \Str:="ERR:-1";
                    ENDIF
                ELSE
                    codigo := ProcesarObjetivo(msg, objetivo);
                    IF codigo = 0 THEN
                        actual := CJointT();
                        objetivo.extax := actual.extax;
                        poseDestino := CalcRobT(objetivo, tool0\WObj:=wobj0);
                        IF poseDestino.trans.z < alturaMinima THEN
                            SocketSend cliSock, \Str:="ERR:-5";
                        ELSE
                            TPWrite "Moviendo por TCP...";
                            MoveAbsJ objetivo, vTcpActual, fine, tool0\WObj:=wobj0;
                            SocketSend cliSock, \Str:="OK";
                        ENDIF
                    ELSE
                        SocketSend cliSock, \Str:="ERR:" + ValToStr(codigo);
                    ENDIF
                ENDIF
            ENDIF
        ENDWHILE

    ERROR (ERR_SOCK_TIMEOUT, ERR_SOCK_CLOSED)
        IF (ERRNO = ERR_SOCK_TIMEOUT) OR (ERRNO = ERR_SOCK_CLOSED) THEN
            SkipWarn;
            IF SocketGetStatus(cliSock) <> SOCKET_CLOSED THEN
                SocketClose cliSock;
            ENDIF
            RETURN;
        ENDIF
    ENDPROC

    LOCAL FUNC num ProcesarObjetivo(string msg, INOUT jointtarget objetivo)
        VAR jointtarget actual;
        VAR bool ok;
        VAR robjoint q;
        VAR robjoint a;

        IF OpMode() <> OP_MAN_PROG THEN
            RETURN -4;
        ENDIF

        ok := StrToVal(msg, q);
        IF ok = FALSE THEN
            RETURN -1;
        ENDIF

        IF Abs(q.rax_1) > 1000 OR Abs(q.rax_2) > 1000 OR Abs(q.rax_3) > 1000 OR
           Abs(q.rax_4) > 1000 OR Abs(q.rax_5) > 1000 OR Abs(q.rax_6) > 1000 THEN
            RETURN -1;
        ENDIF

        IF q.rax_1 < (-165 + margen) OR q.rax_1 > (165 - margen) OR
           q.rax_2 < (-110 + margen) OR q.rax_2 > (110 - margen) OR
           q.rax_3 < (-110 + margen) OR q.rax_3 > (70 - margen) OR
           q.rax_4 < (-160 + margen) OR q.rax_4 > (160 - margen) OR
           q.rax_5 < (-120 + margen) OR q.rax_5 > (120 - margen) OR
           q.rax_6 < (-400 + margen) OR q.rax_6 > (400 - margen) THEN
            RETURN -2;
        ENDIF

        actual := CJointT();
        a := actual.robax;
        IF Abs(q.rax_1 - a.rax_1) > maxPasoDeg OR Abs(q.rax_2 - a.rax_2) > maxPasoDeg OR
           Abs(q.rax_3 - a.rax_3) > maxPasoDeg OR Abs(q.rax_4 - a.rax_4) > maxPasoDeg OR
           Abs(q.rax_5 - a.rax_5) > maxPasoDeg OR Abs(q.rax_6 - a.rax_6) > maxPasoDeg THEN
            RETURN -3;
        ENDIF

        objetivo.robax := q;
        RETURN 0;
    ENDFUNC
ENDMODULE
