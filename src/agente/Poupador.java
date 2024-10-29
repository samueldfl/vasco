package agente;

import java.util.Random;

import algoritmo.ProgramaPoupador;

public class Poupador extends ProgramaPoupador {
    public int acao() {
        Random random = new Random();
        return random.nextInt(6); // Gera um número entre 0 e 5
    }
}
