/*
* Конфигурационный файл содержит строки формата:
*		parameter = value
* Здесь: parameter - имя параметра (не более 24 символов ASCII),
*        value - значение параметра (до 1000 символов ASCII).
* Каждая строка обрабатывается независимо, перенос строки внутри одного выражения недопустим!
* Пробелы и символы табуляции в имени параметра игнорируются,
* пробелы и табуляции в строке значения игнорируются, за исключением текста в двойных кавычках (").
* Пример: строки ниже неразличимы:
*			parameter = value
*			para meter = va lu e
*			parameter = "value"
*			parameter = val "ue"
*		  однако следующая строка не эквивалентна предыдущим:
*			parameter = "va lu e"
* Тем не менее использование пробелов в значениях без двойных кавычек не рекомендуется.
* Двойные кавычки в имени параметра недопустимы!
* Наличие незакрытой двойной кавычки ведёт к ошибке!
* Символ # обозначает начало комментария.
* Весь текст от символа # до конца строки считается комментарием и игнорируется.
* Пример: parameter = value  # comment
* Если параметр отмечен как required, то его отсутствие вызывает ошибку с кодом 4,
* иначе записывается значение параметра, равное "#ND".
*/

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011 4018)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"


int resolve_string(const char* str, char* parameter, char* value)
{
	int i, j, flag, stop, isstring;
	i = j = 0;
	flag = 0;
	stop = 0;
	isstring = 0;
	if (strstr(str, "=") == NULL)
	{
		parameter[0] = '\0';
		flag = 2;
	}
	while (str[i] != '\0')
	{
		switch (str[i])
		{
		case '=':
			parameter[j] = '\0';
			if (flag)
			{
				return 1;
			}
			else
			{
				flag = 1;
			}
			j = 0;
			break;
		case '#':
			stop = 1;
			break;
		case '"':
			if (flag == 1)
			{
				isstring = isstring ? 0 : 1;
			}
			else
			{
				return 1;
			}
			break;
		case '\t':
		case '\n':
		case '\r':
			break;
		case ' ':
			if (!isstring) break;
		default:
			switch (flag)
			{
			case 0:
				parameter[j] = str[i];
				break;
			case 1:
				value[j] = str[i];
				break;
			case 2:
				return 1;
			}
			j++;
		}
		if (stop)
		{
			if (!flag)
			{
				parameter[j] = '\0';
			}
			break;
		}
		i++;
	}
	if (isstring) return 1;
	value[j] = '\0';
	if (!strcmp(parameter, "") && strcmp(value, ""))
	{
		return 1;
	}
	return 0;
}

int read_config(const char* filename, int nparam, const char** parameters, char** values, int* is_required)
{
	int i;
	FILE* fp;
	int* isset;
	int* is_required_loc;
	char buf[1024], param[21], value[1003];
	isset = (int*)malloc(nparam * sizeof(int));
	for (i = 0; i < nparam; i++)
	{
		isset[i] = 0;
	}
	is_required_loc = (int*)malloc(nparam * sizeof(int));
	if (is_required)
	{
		for (i = 0; i < nparam; i++)
		{
			is_required_loc[i] = is_required[i];
		}
	}
	else
	{
		for (i = 0; i < nparam; i++)
		{
			is_required_loc[i] = 0;
		}
	}
	if ((fp = fopen(filename, "r")) == NULL)
	{
		fprintf(stderr, "Error: Cannot open file %s\n", filename);
		return 2;
	}
	while (fgets(buf, 1024, fp))
	{
		if (!resolve_string(buf, param, value))
		{
			for (i = 0; i < nparam; i++)
			{
				if (!strcmp(param, parameters[i]))
				{
					isset[i] = 1;
					strcpy(values[i], value);
					break;
				}
			}
		}
		else
		{
			fprintf(stderr, "Error: Failed to resolve string %d of configuration file:\n%s\n", i, buf);
			return 3;
		}
	}
	fclose(fp);
	for (i = 0; i < nparam; i++)
	{
		if (!isset[i] && is_required_loc[i])
		{
			fprintf(stderr, "Error: Do not have enough data in the configuration file\n");
			return 4;
		}
		if (!isset[i] && !is_required_loc[i])
		{
			values[i][0] = '#';
			values[i][1] = 'N';
			values[i][2] = 'D';
			values[i][3] = '\0';
		}
	}
	free(isset);
	free(is_required_loc);
	return 0;
}

int rewrite_config(const char* filename, int nparam, const char** parameters, const char** values)
{
	int i;
	FILE* out;
	if (!(out = fopen(filename, "w")))
	{
		return 1;
	}
	for (i = 0; i < nparam; i++)
	{
		fprintf(out, "%s= %s\n", parameters[i], values[i]);
	}
	fclose(out);
	return 0;
}
