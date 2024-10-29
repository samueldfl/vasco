package agente;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

import algoritmo.ProgramaLadrao;

public class Ladrao extends ProgramaLadrao {
	private final int PESO_MAX = 20000;
	private final int PESO_MIN = -20000;

	private final int PESO_OLFATO_POUPADOR_UM = 1500;
	private final int PESO_OLFATO_POUPADOR_DOIS = 1300;
	private final int PESO_OLFATO_POUPADOR_TRES = 1000;
	private final int PESO_OLFATO_POUPADOR_QUATRO = 900;
	private final int PESO_OLFATO_POUPADOR_CINCO = 700;

	private final int PESO_PROXIMIDADE_CURTA = 9400;
	private final int PESO_PROXIMIDADE_LONGA = 8340;

	private final int[][] POSICOES_LONGAS = {
			{ 0, 1, 2, 3, 4 },
			{ 5, 9 },
			{ 10, 13 },
			{ 14, 18 },
			{ 19, 20, 21, 22, 23 }
	};
	private final int POSICAO_CIMA = 7;
	private final int POSICAO_BAIXO = 16;
	private final int POSICAO_ESQUERDA = 11;
	private final int POSICAO_DIREITA = 12;
	private final int POSICAO_DIAGONAL_CURTA_CIMA_ESQUERDA = 6;
	private final int POSICAO_DIAGONAL_CURTA_CIMA_DIREITA = 8;
	private final int POSICAO_DIAGONAL_CURTA_BAIXO_ESQUERDA = 15;
	private final int POSICAO_DIAGONAL_CURTA_BAIXO_DIREITA = 17;

	private final int VISAO_LOCAL_INDISPONIVEL = -2;
	private final int VISAO_INDISPONIVEL = -1;
	private final int VISAO_PAREDE = 1;
	private final int VISAO_BANCO = 3;
	private final int VISAO_MOEDA = 4;
	private final int VISAO_PODER = 5;

	private final int MOVIMENTO_CIMA = 1;
	private final int MOVIMENTO_BAIXO = 2;
	private final int MOVIMENTO_DIREITA = 3;
	private final int MOVIMENTO_ESQUERDA = 4;

	private int PESO_CIMA = 1;
	private int PESO_BAIXO = 1;
	private int PESO_DIREITA = 1;
	private int PESO_ESQUERDA = 1;

	static ArrayList<Point> historico = new ArrayList<>();

	public int acao() {
		PESO_CIMA = 1;
		PESO_BAIXO = 1;
		PESO_DIREITA = 1;
		PESO_ESQUERDA = 1;

		int[] visao = sensor.getVisaoIdentificacao();

		heuristicaExplorar(sensor.getPosicao(), visao);

		heuristicaMovimentosInuteis(visao);
		heuristicaVisaoCurta(visao);
		heuristicaVisaoLonga(visao);

		int[] olfatoPoupador = sensor.getAmbienteOlfatoPoupador();

		heuristicaOlfato(olfatoPoupador);

		return agir();
	}

	private void heuristicaExplorar(final Point posicao, final int[] visao) {
		int[][] direcoes = {
				{ posicao.x, posicao.y - 1 },
				{ posicao.x, posicao.y + 1 },
				{ posicao.x - 1, posicao.y },
				{ posicao.x + 1, posicao.y }
		};

		int[] pesos = { PESO_CIMA, PESO_BAIXO, PESO_ESQUERDA, PESO_DIREITA };
		int pesoNaoVisitado = 7000;
		int pesoVisitaRecente = -3000;

		ArrayList<Integer> direcoesRecentementeVisitadas = new ArrayList<>();

		for (int i = 0; i < direcoes.length; i++) {
			Point posicaoAdjacente = new Point(direcoes[i][0], direcoes[i][1]);
			boolean jaVisitado = false;

			for (Point ponto : historico) {
				if (ponto.equals(posicaoAdjacente)) {
					jaVisitado = true;
				}
			}

			if (!movimentoInutil(visao[i])) {
				if (!jaVisitado) {
					pesos[i] += pesoNaoVisitado;
				} else {
					direcoesRecentementeVisitadas.add(i);
				}
			}
		}

		for (int direcao : direcoesRecentementeVisitadas) {
			pesos[direcao] += pesoVisitaRecente;
		}

		atualizarPesos(pesos);

		historico.add(new Point(posicao.x, posicao.y));
		if (historico.size() > 150) {
			historico.remove(0);
		}
	}

	private void heuristicaOlfato(final int[] olfato) {
		for (int i = 0; i < olfato.length; i++) {
			int distancia = olfato[i];

			switch (i) {
				case 0 -> {
					atribuirPesoOlfatoPoupador(POSICAO_CIMA, distancia);
					atribuirPesoOlfatoPoupador(POSICAO_ESQUERDA, distancia);
				}
				case 1 -> {
					atribuirPesoOlfatoPoupador(POSICAO_CIMA, distancia);
				}
				case 2 -> {
					atribuirPesoOlfatoPoupador(POSICAO_CIMA, distancia);
					atribuirPesoOlfatoPoupador(POSICAO_DIREITA, distancia);
				}
				case 3 -> {
					if (!movimentoInutil(POSICAO_ESQUERDA)) {
						atribuirPesoOlfatoPoupador(POSICAO_ESQUERDA, distancia);
					}
				}
				case 4 -> {
					atribuirPesoOlfatoPoupador(POSICAO_DIREITA, distancia);
				}
				case 5 -> {
					atribuirPesoOlfatoPoupador(POSICAO_BAIXO, distancia);
					atribuirPesoOlfatoPoupador(POSICAO_ESQUERDA, distancia);
				}
				case 6 -> {
					atribuirPesoOlfatoPoupador(POSICAO_BAIXO, distancia);
				}
				case 7 -> {
					atribuirPesoOlfatoPoupador(POSICAO_BAIXO, distancia);
					atribuirPesoOlfatoPoupador(POSICAO_DIREITA, distancia);
				}
			}
		}
	}

	private void atribuirPesoOlfatoPoupador(final int posicao, final int distancia) {
		int peso = obterPesoPorDistancia(distancia);

		if (peso > 0) {
			adicionarPesoPorPosicao(posicao, peso);
		}
	}

	private int obterPesoPorDistancia(final int distancia) {
		switch (distancia) {
			case 1:
				return PESO_OLFATO_POUPADOR_UM;
			case 2:
				return PESO_OLFATO_POUPADOR_DOIS;
			case 3:
				return PESO_OLFATO_POUPADOR_TRES;
			case 4:
				return PESO_OLFATO_POUPADOR_QUATRO;
			case 5:
				return PESO_OLFATO_POUPADOR_CINCO;
			default:
				return 0;
		}
	}

	private void adicionarPesoPorPosicao(final int posicao, final int peso) {
		if (posicao == POSICAO_BAIXO) {
			PESO_BAIXO += peso;
		} else if (posicao == POSICAO_CIMA) {
			PESO_CIMA += peso;
		} else if (posicao == POSICAO_ESQUERDA) {
			PESO_ESQUERDA += peso;
		} else if (posicao == POSICAO_DIREITA) {
			PESO_DIREITA += peso;
		}
	}

	private void heuristicaMovimentosInuteis(int[] visao) {
		int[] posicoes = { POSICAO_CIMA, POSICAO_BAIXO, POSICAO_DIREITA, POSICAO_ESQUERDA };

		for (int posicao : posicoes) {
			if (movimentoInutil(visao[posicao])) {
				atribuirPeso(posicao, PESO_MIN);
			}
		}
	}

	private boolean movimentoInutil(int posicao) {
		return posicao == VISAO_PAREDE || posicao == VISAO_MOEDA || posicao == VISAO_PODER
				|| posicao == VISAO_INDISPONIVEL || posicao == VISAO_BANCO
				|| posicao == VISAO_LOCAL_INDISPONIVEL || (posicao >= 200 && posicao <= 299);
	}

	private void atribuirPeso(int posicao, int valor) {
		int[] pesos = { PESO_CIMA, PESO_BAIXO, PESO_ESQUERDA, PESO_DIREITA };
		int indice = obterIndicePorPosicao(posicao);

		if (indice != -1) {
			pesos[indice] += valor;

			atualizarPesos(pesos);
		}
	}

	private int obterIndicePorPosicao(int posicao) {
		switch (posicao) {
			case POSICAO_CIMA:
				return 0;
			case POSICAO_BAIXO:
				return 1;
			case POSICAO_ESQUERDA:
				return 2;
			case POSICAO_DIREITA:
				return 3;
			default:
				return -1;
		}
	}

	private void atualizarPesos(int[] pesos) {
		PESO_CIMA = pesos[0];
		PESO_BAIXO = pesos[1];
		PESO_ESQUERDA = pesos[2];
		PESO_DIREITA = pesos[3];
	}

	private void heuristicaVisaoCurta(int[] visao) {
		int valorPosicao = visao[POSICAO_DIAGONAL_CURTA_CIMA_ESQUERDA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_CIMA])) {
				atribuirPeso(POSICAO_CIMA, PESO_PROXIMIDADE_CURTA);
			}

			if (!movimentoInutil(visao[POSICAO_ESQUERDA])) {
				atribuirPeso(POSICAO_ESQUERDA, PESO_PROXIMIDADE_CURTA);
			}
		}

		valorPosicao = visao[POSICAO_CIMA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_CIMA])) {
				atribuirPeso(POSICAO_CIMA, PESO_MAX);
			}
		}

		valorPosicao = visao[POSICAO_DIAGONAL_CURTA_CIMA_DIREITA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_CIMA])) {
				atribuirPeso(POSICAO_CIMA, PESO_PROXIMIDADE_CURTA);
			}

			if (!movimentoInutil(visao[POSICAO_DIREITA])) {
				atribuirPeso(POSICAO_DIREITA, PESO_PROXIMIDADE_CURTA);
			}
		}

		valorPosicao = visao[POSICAO_DIAGONAL_CURTA_BAIXO_ESQUERDA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_BAIXO])) {
				atribuirPeso(POSICAO_BAIXO, PESO_PROXIMIDADE_CURTA);
			}

			if (!movimentoInutil(visao[POSICAO_ESQUERDA])) {
				atribuirPeso(POSICAO_ESQUERDA, PESO_PROXIMIDADE_CURTA);
			}
		}

		valorPosicao = visao[POSICAO_BAIXO];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_BAIXO])) {
				atribuirPeso(POSICAO_BAIXO, PESO_MAX);
			}
		}

		valorPosicao = visao[POSICAO_DIAGONAL_CURTA_BAIXO_DIREITA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_BAIXO])) {
				atribuirPeso(POSICAO_BAIXO, PESO_PROXIMIDADE_CURTA);
			}

			if (!movimentoInutil(visao[POSICAO_DIREITA])) {
				atribuirPeso(POSICAO_DIREITA, PESO_PROXIMIDADE_CURTA);
			}
		}

		valorPosicao = visao[POSICAO_DIREITA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_DIREITA])) {
				atribuirPeso(POSICAO_DIREITA, PESO_MAX);
			}
		}

		valorPosicao = visao[POSICAO_ESQUERDA];
		if (valorPosicao >= 100 && valorPosicao <= 199) {
			if (!movimentoInutil(visao[POSICAO_ESQUERDA])) {
				atribuirPeso(POSICAO_ESQUERDA, PESO_MAX);
			}
		}
	}

	private void heuristicaVisaoLonga(int[] visao) {
		for (int i = 0; i < POSICOES_LONGAS.length; i++) {
			for (int posicao : POSICOES_LONGAS[i]) {
				int valorPosicao = visao[posicao];

				if (valorPosicao >= 100 && valorPosicao <= 199) {
					switch (posicao) {
						case 0, 1, 5 -> {

							if (!movimentoInutil(visao[POSICAO_ESQUERDA])) {
								atribuirPeso(POSICAO_ESQUERDA, PESO_PROXIMIDADE_LONGA);
							}
							if (!movimentoInutil(visao[POSICAO_CIMA])) {
								atribuirPeso(POSICAO_CIMA, PESO_PROXIMIDADE_LONGA);
							}
						}
						case 2 -> {

							if (!movimentoInutil(visao[POSICAO_CIMA])) {
								atribuirPeso(POSICAO_CIMA, PESO_PROXIMIDADE_LONGA);
							}
						}
						case 3, 4, 9 -> {
							if (!movimentoInutil(POSICAO_CIMA)) {
								atribuirPeso(POSICAO_CIMA, PESO_PROXIMIDADE_LONGA);
							}
							if (!movimentoInutil(POSICAO_DIREITA)) {
								atribuirPeso(POSICAO_CIMA, PESO_PROXIMIDADE_LONGA);
							}
						}
						case 13 -> {
							if (!movimentoInutil(visao[POSICAO_DIREITA])) {
								atribuirPeso(POSICAO_DIREITA, PESO_PROXIMIDADE_LONGA);
							}
						}
						case 18, 22, 23 -> {
							if (!movimentoInutil(visao[POSICAO_BAIXO])) {
								atribuirPeso(POSICAO_BAIXO, PESO_PROXIMIDADE_LONGA);
							}
							if (!movimentoInutil(visao[POSICAO_ESQUERDA])) {
								atribuirPeso(POSICAO_ESQUERDA, PESO_PROXIMIDADE_LONGA);
							}
						}
						case 21 -> {
							if (!movimentoInutil(visao[POSICAO_BAIXO])) {
								atribuirPeso(POSICAO_BAIXO, PESO_PROXIMIDADE_LONGA);
							}
						}
						case 14, 19, 20 -> {
							if (!movimentoInutil(visao[POSICAO_DIREITA])) {
								atribuirPeso(POSICAO_DIREITA, PESO_PROXIMIDADE_LONGA);
							}
						}
					}
				}
			}
		}
	}

	private int agir() {
		if (PESO_BAIXO == PESO_MIN && PESO_CIMA == PESO_MIN && PESO_DIREITA == PESO_MIN && PESO_ESQUERDA == PESO_MIN) {
			return 0;
		}

		int maiorPeso = Math.max(Math.max(PESO_BAIXO, PESO_CIMA), Math.max(PESO_DIREITA, PESO_ESQUERDA));

		List<Integer> acoesComMaiorPeso = new ArrayList<>();

		if (PESO_BAIXO == maiorPeso)
			acoesComMaiorPeso.add(MOVIMENTO_BAIXO);

		if (PESO_CIMA == maiorPeso)
			acoesComMaiorPeso.add(MOVIMENTO_CIMA);

		if (PESO_DIREITA == maiorPeso)
			acoesComMaiorPeso.add(MOVIMENTO_DIREITA);

		if (PESO_ESQUERDA == maiorPeso)
			acoesComMaiorPeso.add(MOVIMENTO_ESQUERDA);

		int resultado = acoesComMaiorPeso.get((int) (Math.random() * acoesComMaiorPeso.size()));
		return resultado;
	}
}
