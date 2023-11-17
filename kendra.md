# Kendra 성능 향상 방법

Kendra의 [Retrieve API](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)를 이용합니다.

## Retrieve API

[Retrieve](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)는 Default Quota 기준으로 200 개의 token으로 된 단어 100개의  

## Type 

Type의 종류에는 "DOCUMENT", "QUESTION_ANSWER", "ANSWER"가 있습니다.

Type은 query결과에서 "Format"으로 구분할 수 있습니다.

## Score

[ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)와 같이 "VERY_HIGH", "HIGH", "MEDIUM", "LOW", "NOT_AVAILABLE"로 결과의 신뢰도를 확인할 수 있습니다.

```text
[
   {
      "Id":"74719041-8126-473c-92f1-929fdc520138-188b319d-552f-4ff4-a7d5-8cbcd21dbea8",
      "Type":"QUESTION_ANSWER",
      "Format":"TEXT",
      "AdditionalAttributes":[
         {
            "Key":"QuestionText",
            "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
            "Value":{
               "TextWithHighlightsValue":{
                  "Text":"How many free clinics are in Spokane WA?",
                  "Highlights":[
                     {
                        "BeginOffset":4,
                        "EndOffset":8,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":9,
                        "EndOffset":13,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":14,
                        "EndOffset":21,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":29,
                        "EndOffset":36,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     },
                     {
                        "BeginOffset":37,
                        "EndOffset":39,
                        "TopAnswer":false,
                        "Type":"STANDARD"
                     }
                  ]
               }
            }
         },
         {
            "Key":"AnswerText",
            "ValueType":"TEXT_WITH_HIGHLIGHTS_VALUE",
            "Value":{
               "TextWithHighlightsValue":{
                  "Text":"13",
                  "Highlights":[
                     
                  ]
               }
            }
         }
      ],
      "DocumentId":"c24c0fe9cbdfa412ac58d1b5fc07dfd4afd21cbd0f71df499f305296d985a8c9a91f1b2c-e28b-4d13-8b01-8a33be5fc126",
      "DocumentTitle":{
         "Text":""
      },
      "DocumentExcerpt":{
         "Text":"13",
         "Highlights":[
            {
               "BeginOffset":0,
               "EndOffset":2,
               "TopAnswer":false,
               "Type":"STANDARD"
            }
         ]
      },
      "DocumentURI":"https://www.freeclinics.com/",
      "DocumentAttributes":[
         {
            "Key":"_source_uri",
            "Value":{
               "StringValue":"https://www.freeclinics.com/"
            }
         }
      ],
      "ScoreAttributes":{
         "ScoreConfidence":"VERY_HIGH"
      },
      "FeedbackToken":"AYADeN-jZ9DvGVP9n00b4d48LrsAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREF6ajZZVkJ3M3B4dXZEMGRJZitQaEEzUWNVZkE3TDVBbjNEOCs1bE1aRm1hN1M3a0N3cjNiMzZRR2hPcTloeVJ1QT09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwEP6KAczOsL3xpUp6oizSAgAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMT_QgV_BMu5l49EZAAgEQgDuTOkP1QJbt85KZ4FDF438i0upluDZq_Rf3L8H9PqkLQOSgUAyyy9hqEmMOZUcGqvBNc_ekw4pbMRy5ZAIAAAAADAAAEAAAAAAAAAAAAAAAAAB3LYEFCQuAEb8NnKHSk1eT_____wAAAAEAAAAAAAAAAAAAAAEAAAF2QLnDNTO_Ma1EGreEOHC8YG5ijJ7jLblLE4CbyAY9ueJHKBTQ-Rf2A_pD9hpXTuyP6Ho84IIlScm7IhFUomBUSZMD_qrc0qnvlrCjgXwJ_AM0MJKmqBkMPNvivFnfZ9xl-dFyFX1sdzq0_LUE4KgLZpjQiSU0b_PFJw2zN8P6JSJb9Fz84fbWu1_nzrVrqCj5dDpMLDNLgC3f6pTS4IqmJqMsj6BbGcdsvLIzVA2XaAGYS8CNv9pu5Hz63yrh6hG4UHWJwdhIcPZG7z7BayFjravsKjw101PJnzUKSIfiZlRnoqm-Bbff-ieECV-vZ_1vtskbHhmsZ4WlKTcpD5QGMrElbk7WMbdPf8gmGQfC8SMrR-ixO7d856LIsoTx9i6VcN91GxEKcYtsXY4J0w6G4aL8-tj1iS7zwIsxHimIsuAHM4u5SmHmI_oJ25pR-7TA2K34GVv9VhYydG8JsBbjGV-mPpg6ORE4bNkhRL38f1pfHEXNlv79F9b8UP93MLfj6lZT25tPAGcwZQIxAK_GncCyOyt2NLdszY-Oc2Qchpo2CCTjj25a5wyzYv4JObw591oaxZeSVbA_Mq2v2gIwf6xq0c5vYGait9J9mnI2FMtEJ3rI2DRld30IRmWWNES54XOxciMd5J_YxJGkfghX.74719041-8126-473c-92f1-929fdc520138-188b319d-552f-4ff4-a7d5-8cbcd21dbea8"
   }
]
```

## FAQ

[FAQ-Kendra](https://github.com/aws-samples/enterprise-search-with-amazon-kendra-workshop/blob/master/Part%202%20-%20Adding%20a%20FAQ.md)를 참조합니다.

[kendra-faq-refresher](https://github.com/aws-samples/amazon-kendra-faq-refresher/tree/main)를 참조하여 FAQ를 Kendra 검색 결과로 활용할 수 있습니다.


## Reference

[Retrieve API](https://docs.aws.amazon.com/kendra/latest/APIReference/API_Retrieve.html)

## Document

```text
{
   "QueryId":"cab6a783-5daf-40b6-a3d9-4df51f3ec812",
   "ResultItems":[
      {
         "Id":"cab6a783-5daf-40b6-a3d9-4df51f3ec812-406002d2-7bd5-4dca-a2f7-2edceecd70a6",
         "Type":"DOCUMENT",
         "Format":"TEXT",
         "AdditionalAttributes":[
            
         ],
         "DocumentId":"4fc78d8c-b9d8-4abb-8e8c-c99523e2f1d5",
         "DocumentTitle":{
            "Text":"fsi_faq_ko.csv",
            "Highlights":[
               
            ]
         },
         "DocumentExcerpt":{
            "Text":"...인증서는 어디서 발급 하나요?\t인증서는 서울은행에 인터넷뱅킹서비스가 가입되어 있으시면 서울은행 홈페이지 또는 서울 모바일앱의 인증센터에서 발급 가능합니다. ※ 개인은 주민번호로 금융기관 통합하여 발급기관별, 용도별로 한 개의 인증서만 발급이 가능합니다. 예) 타은행에서 은행/신용카드/보험용(결제원) 인증서를 발급받은 경우 → 당행 은행/신용카드/보험용(결제원) 인증서 발급 불가\t인터넷뱅킹\t서울은행\n\t개인정보 변경은 어디서 해야 하나요...",
            "Highlights":[
               
            ]
         },
         "DocumentURI":"",
         "DocumentAttributes":[
            
         ],
         "ScoreAttributes":{
            "ScoreConfidence":"MEDIUM"
         },
         "FeedbackToken":"AYADeEIT0ORSX08Sv-t8hqDFj2UAXwABABVhd3MtY3J5cHRvLXB1YmxpYy1rZXkAREF1VzA4b1FGeEIwbDhVL0R2Y2s3RllvVHphcXI5ZG1hVVdMVU5hVWl2RllRaW5ZeUg5SDdQbC8zVlYwNmJuRUN0UT09AAEAB2F3cy1rbXMAS2Fybjphd3M6a21zOnVzLXdlc3QtMjoxNDk0MDA5NDM5NTk6a2V5LzUyN2YwMjRhLTUyMDktNDI4NC1iOTYwLTJhMjYxMzQxNWNkNgC4AQIBAHhoFIrDBc0sA_W0qqJvieboGJWYBK_hEm739PftPtfwZwHBaXKzCFbOpmHzKOt9tucTAAAAfjB8BgkqhkiG9w0BBwagbzBtAgEAMGgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMj2BQ8y0RoHFXVV1AAgEQgDtMdfOimJQuRZVIhoS9xaeORWplKcm2xX_spfBtv10z2USf9Wl3xSV_1twmqnJ7WWh3BgCXlqcCylQZPgIAAAAADAAAEAAAAAAAAAAAAAAAAABnOJofpWuq2r-4USU4fu9A_____wAAAAEAAAAAAAAAAAAAAAEAAAF24b73mqfm5Unxu25vRMlukIUI1i3trLzFlnyfCvCin8J7GsN3pu25FiI0Hu3VLAXusFfik00Gp7kzT8ZcFVUgp0m9FZXeG2SlFkTssFxIn8QLWPeW1ea8TgZr0eeCk6bjmtVfQSjv5JP8Gk2Trk0cC2D20jH_aWsZ1Ri78bY5H0NOsS2LsV1CHEkDacfNIyhwMaiDpYWyn8sCIV2a7OmkdZOwTkEutbD69McCISGGhlh5c8ZFJaXESjtJCZuPf4EFQHH5oIBHSNzivaXD4tGGzSa1XabMD36rvYqYykbG09UE12eRB-Gvy8BuEaPIsAP0Ib5rPBNbTImCiw10nDm4oTnB9LsW6lPT-x_DI8qoxEdrj0sH5WjdQKhvEt1g33-6w0AlIQNMiNaw-CziK-PgK8YQFlnL2HLPdmPCZBCH_W18nM2oyAMjAzySgq3KCExKyaqylFn5DsPXLjh46J60ZMxf1DhQ9U82DiQCuJ5G1Z9rxdMASmRHWFSOKxXolyMwTCShOI5vAGcwZQIxAKd5oE-KmuivvmQmDQl-PtQRzhpWaUWthnu-NP6serOPYvVjmYdxNvNtbyZy-3pJ_AIwLszR3dnL6GPu_ed2QlY6r2yVs_l-O3hNKB7WMv8P23cajHEX8qNbr_I_yafIFFoG.cab6a783-5daf-40b6-a3d9-4df51f3ec812-406002d2-7bd5-4dca-a2f7-2edceecd70a6"
      },
   ],
   "FacetResults":[
      
   ],
   "TotalNumberOfResults":4,
   "ResponseMetadata":{
      "RequestId":"52b5d039-a600-4ed8-90b1-d1fa46c5e1c7",
      "HTTPStatusCode":200,
      "HTTPHeaders":{
         "x-amzn-requestid":"52b5d039-a600-4ed8-90b1-d1fa46c5e1c7",
         "content-type":"application/x-amz-json-1.1",
         "content-length":"9569",
         "date":"Fri, 17 Nov 2023 06:55:10 GMT"
      },
      "RetryAttempts":0
   }
}
```
